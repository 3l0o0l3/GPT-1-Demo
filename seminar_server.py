"""Anonymous frozen-model audience server with a separate loopback presenter API."""
import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import hashlib
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from seminar_store import SeminarError, SeminarStore

LIMITS = dict(messageChars=800, maxTurns=10, maxNewTokens=160, windowSize=64, temperature=0.8)


class RateLimits:
    def __init__(self, clock=time.monotonic):
        self.clock, self.items, self.lock = clock, {}, threading.Lock()

    def check(self, key, limit):
        with self.lock:
            now = self.clock()
            if len(self.items) >= 10000 and key not in self.items:
                self.items = {k: v for k,v in self.items.items() if v and v[-1] > now-60}
                if len(self.items) >= 10000:
                    raise SeminarError(429, 'rate_limit', 'The seminar is busy. Please try again shortly.')
            hits = self.items.setdefault(key, deque())
            while hits and hits[0] <= now-60:
                hits.popleft()
            if len(hits) >= limit:
                raise SeminarError(429, 'rate_limit', 'Please wait a minute before trying again.')
            hits.append(now)


class AudienceApplication:
    def __init__(self, store, model, public_url, static_dir, share=None, publisher=None):
        self.share = share
        self.publisher = publisher
        public_url = public_url or 'https://closed.invalid/seminar-gpt/'
        parsed = urlsplit(public_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('Provide an absolute public HTTP(S) URL without credentials or query.')
        self.store, self._model = store, model
        self.public_url = public_url.rstrip('/')+'/'
        self.origin = f'{parsed.scheme}://{parsed.netloc}'
        self.host, self.prefix, self.secure = parsed.netloc, parsed.path.rstrip('/'), parsed.scheme == 'https'
        self.static_dir = Path(static_dir)
        self.rates = RateLimits()
        self.slots, self.worker = threading.BoundedSemaphore(16), threading.Lock()

    @property
    def model(self):
        return self.publisher.current_model if self.publisher is not None else self._model

    def model_publication(self):
        if self.publisher is not None:
            return self.publisher.status()
        return {'enabled': False, 'releaseId': None, 'publishedAt': None, 'lastRequestId': None}

    def publication_info(self):
        if self.publisher is not None:
            return self.publisher.describe()
        return {'model': self.model.metadata(), 'modelPublication': self.model_publication()}

    def info(self, access=None):
        result = dict(model=self.model.metadata(), limits=dict(LIMITS),
                      publicUrl=access['publicUrl'] if access else self.public_url)
        if self.share:
            result['shareExpiresAt'] = self.share.status()['expiresAt']
        return result

    @staticmethod
    def csrf(token):
        return hashlib.sha256(('seminar-csrf:'+token).encode()).hexdigest()

    def session_response(self, token, conversation, access=None):
        return dict(self.info(access), csrf=self.csrf(token), conversation=conversation)

    @staticmethod
    def prompt(turn):
        messages = [message for message in turn['messages'] if message.get('kind') != 'classification']
        if not messages:
            return turn['userMessage']['content']
        history = '\n'.join(('User: ' if m['role']=='user' else 'Model: ')+m['content'] for m in messages)
        # Whole visible transcript remains stored; inference only sees a short suffix.
        return (history+'\nUser: '+turn['userMessage']['content']+'\nModel:')[-19000:]


class AudienceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, app, private=False):
        self.app, self.private = app, private
        self.connections = threading.BoundedSemaphore(64)
        super().__init__(address, AudienceHandler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class AudienceHandler(BaseHTTPRequestHandler):
    server_version = 'SeminarGPT/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(40)
        self.streaming = False
        self.access = None

    def _check_access(self):
        if self.server.app.share:
            self.server.app.share.check(self.access['id'])

    def _access_guard(self):
        return self.server.app.share.guard(self.access['id']) if self.server.app.share else nullcontext()

    def log_message(self, format, *args):
        pass  # Never log chat text, session cookies, or request IDs.

    def _headers(self, status, kind, length=None, cookie=None):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if length is not None:
            self.send_header('Content-Length', str(length))
        if cookie:
            self.send_header('Set-Cookie', cookie)
        self.send_header('Connection', 'close')
        self.close_connection = True
        self.end_headers()

    def _json(self, status, value, cookie=None):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, 'application/json; charset=utf-8', len(body), cookie)
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _event(self, event):
        if not self.streaming:
            self._headers(200, 'application/x-ndjson; charset=utf-8')
            self.streaming = True
        self.wfile.write((json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n').encode())
        self.wfile.flush()

    def _body(self):
        if (self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1 or
                self.headers.get('Content-Type', '').split(';')[0] != 'application/json'):
            raise SeminarError(400, 'invalid_input', 'Send a bounded JSON request.')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if not 0 < length <= 16384:
            raise SeminarError(413, 'body_limit', 'The request is too large or empty.')
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError()
            value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError):
            raise SeminarError(400, 'invalid_input', 'The request must be a JSON object.')
        if not isinstance(value, dict):
            raise SeminarError(400, 'invalid_input', 'The request must be a JSON object.')
        return value

    def _token(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
            token = cookie['seminar_session'].value if 'seminar_session' in cookie else None
        except Exception:
            token = None
        return token if token and len(token) == 43 and all(c.isalnum() or c in '-_' for c in token) else None

    def _authenticate(self):
        token = self._token()
        if not token or not hmac.compare_digest(self.headers.get('X-CSRF-Token',''), self.server.app.csrf(token)):
            raise SeminarError(403, 'csrf', 'Reload the page before trying again.')
        self.server.app.store.current(token)
        return token

    @staticmethod
    def _fields(body, expected):
        if set(body) != set(expected):
            raise SeminarError(400, 'invalid_input', 'Unexpected or missing request fields.')

    @staticmethod
    def _text(value, maximum, empty=False):
        if not isinstance(value, str) or len(value)>maximum or (not empty and not value.strip()):
            raise SeminarError(400, 'invalid_input', f'Text must contain {0 if empty else 1}–{maximum} characters.')
        return value.strip()

    def _private(self, parsed):
        app = self.server.app
        if (self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}') or
                self.headers.get('Origin') or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            raise SeminarError(403, 'invalid_host', 'Use the private presenter gateway.')
        if self.command == 'POST' and parsed.path == '/admin/model/sync' and not parsed.query:
            body = self._body()
            if app.publisher is None:
                raise SeminarError(503, 'publication_disabled', 'Model updates are not configured.')
            return self._json(200, app.publisher.sync(body, app.worker))
        if self.command == 'POST' and parsed.path == '/admin/submissions/clear' and not parsed.query:
            body = self._body()
            self._fields(body, ('throughCursor',))
            return self._json(200, app.store.clear_submissions(body['throughCursor']))
        if self.command == 'POST' and not parsed.query and parsed.path in ('/admin/share', '/admin/share/revoke'):
            if app.share is None:
                raise SeminarError(503, 'share_unavailable', 'Temporary sharing is not configured.')
            body = self._body()
            if parsed.path == '/admin/share':
                self._fields(body, ('durationMinutes',))
                result = app.share.create(body['durationMinutes'])
            else:
                self._fields(body, ())
                result = app.share.revoke()
            return self._json(200, {'share': result})
        if self.command == 'GET' and parsed.path == '/admin/status' and not parsed.query:
            result = dict(publicUrl=app.public_url, **app.publication_info(), **app.store.inbox_status())
            if app.share:
                result['share'] = app.share.status()
                result['publicUrl'] = result['share']['publicUrl']
            return self._json(200, result)
        if self.command == 'GET' and parsed.path == '/admin/submissions':
            q = parse_qs(parsed.query, keep_blank_values=True)
            if set(q)-{'after','limit'} or any(len(v)!=1 for v in q.values()):
                raise SeminarError(400, 'invalid_input', 'Invalid pagination.')
            try:
                after, limit = int(q.get('after',['0'])[0]), int(q.get('limit',['100'])[0])
                if after < 0 or not 1 <= limit <= 100:
                    raise ValueError()
            except ValueError:
                raise SeminarError(400, 'invalid_input', 'Invalid pagination.')
            return self._json(200, app.store.submissions(after, limit))
        raise SeminarError(404, 'not_found', 'Not found.')

    def _chat(self, token, body):
        app = self.server.app
        self._fields(body, ('message','requestId'))
        content = self._text(body['message'], 800)
        try:
            request_id = str(uuid.UUID(body['requestId']))
        except (ValueError, TypeError, AttributeError):
            raise SeminarError(400, 'invalid_input', 'A valid request ID is required.')
        app.rates.check('visitor:'+app.store.digest(token), 12)
        if not app.slots.acquire(blocking=False):
            raise SeminarError(429, 'queue_full', 'The model queue is full. Please try again shortly.')
        turn, locked = None, False
        try:
            turn = app.store.begin_turn(token, request_id, content)
            self._event({'type':'queued'})
            locked = app.worker.acquire(timeout=30)
            if not locked:
                raise SeminarError(503, 'queue_timeout', 'The model is busy. Your message was not completed; please try again.')
            self._check_access()
            provider = app.model
            metadata = provider.metadata()
            self._event({'type':'start', 'userMessage':turn['userMessage'], 'model':metadata})

            def receive(event):
                if event['type'] == 'token':
                    self._check_access()
                    self._event({'type':'token', 'index':event['index'], 'content':event['continuation']})

            result = provider.generate(app.prompt(turn), max_new_tokens=160, temperature=.8,
                                        seed=secrets.randbelow(2147483648), on_event=receive)
            self._check_access()
            with self._access_guard():
                conversation, assistant = app.store.complete_turn(turn, result['continuation'], metadata,
                                                                  result['stopReason'], len(result['tokens']))
            turn = None
            self._event({'type':'done', 'conversation':conversation, 'assistantMessage':assistant, 'model':metadata})
        finally:
            if locked:
                app.worker.release()
            if turn is not None:
                app.store.abort_turn(turn)
            app.slots.release()

    def _classify(self, token, body):
        app = self.server.app
        self._fields(body, ('message', 'requestId'))
        content = self._text(body['message'], 800)
        try:
            request_id = str(uuid.UUID(body['requestId']))
        except (ValueError, TypeError, AttributeError):
            raise SeminarError(400, 'invalid_input', 'A valid request ID is required.')
        if not app.model.metadata().get('classifier', {}).get('available'):
            raise SeminarError(503, 'classifier_unavailable', 'No trained classifier has been published yet.')
        app.rates.check('visitor:' + app.store.digest(token), 12)
        if not app.slots.acquire(blocking=False):
            raise SeminarError(429, 'queue_full', 'The model queue is full. Please try again shortly.')
        turn, locked = None, False
        try:
            turn = app.store.begin_turn(token, request_id, content)
            locked = app.worker.acquire(timeout=30)
            if not locked:
                raise SeminarError(503, 'queue_timeout', 'The model is busy. Recover your conversation before retrying.')
            self._check_access()
            provider = app.model
            metadata = provider.metadata()
            if not metadata.get('classifier', {}).get('available'):
                raise SeminarError(503, 'classifier_unavailable', 'No trained classifier has been published yet.')
            result = provider.classify(content)
            summary = '\n'.join(
                ('Pretrained → sentiment' if item['branch'] == 'pretrained' else 'Random → sentiment')
                + ': ' + item['prediction'] + ' (' + ', '.join(
                    f"{p['label']} {p['p']:.1%}" for p in item['probabilities']) + ')'
                for item in result['branches'])
            with self._access_guard():
                conversation, assistant = app.store.complete_turn(turn, summary, metadata,
                    'classification', 0, classification=result)
            turn = None
            return self._json(200, {'conversation': conversation, 'assistantMessage': assistant, 'model': metadata})
        finally:
            if locked:
                app.worker.release()
            if turn is not None:
                app.store.abort_turn(turn)
            app.slots.release()

    def _run(self):
        app = self.server.app
        parsed = urlsplit(self.path)
        if self.server.private:
            return self._private(parsed)
        if app.share:
            self.access = app.share.resolve(self.headers.get('Host'), parsed.path, self.headers.get('Origin'))
        else:
            self.access = dict(id=None, publicUrl=app.public_url, origin=app.origin,
                               host=app.host, prefix=app.prefix, secure=app.secure)
        access = self.access
        if self.headers.get('Host') != access['host']:
            raise SeminarError(403, 'invalid_host', 'Use the seminar public URL.')
        origin = self.headers.get('Origin')
        if origin and origin != access['origin']:
            raise SeminarError(403, 'invalid_origin', 'Use the seminar page to send requests.')
        path = parsed.path
        if path == access['prefix'] and self.command in ('GET','HEAD'):
            self.send_response(308)
            self.send_header('Location', access['publicUrl'])
            self.send_header('Content-Length','0')
            self.end_headers()
            return
        if not path.startswith(access['prefix']+'/'):
            raise SeminarError(404, 'not_found', 'Not found.')
        path = path[len(access['prefix']):]
        if self.command in ('GET','HEAD'):
            assets = {'/':('index.html','text/html; charset=utf-8'), '/app.js':('app.js','text/javascript; charset=utf-8'), '/style.css':('style.css','text/css; charset=utf-8')}
            if path in assets:
                filename, kind = assets[path]
                body = (app.static_dir/filename).read_bytes()
                self._headers(200, kind, len(body))
                if self.command != 'HEAD':
                    self.wfile.write(body)
                return
            if path == '/api/info' and not parsed.query:
                return self._json(200, app.info(access))
            raise SeminarError(404, 'not_found', 'Not found.')
        if self.command != 'POST' or parsed.query or path not in ('/api/session','/api/new','/api/chat','/api/classify','/api/submit'):
            raise SeminarError(404, 'not_found', 'Not found.')
        body = self._body()
        self._check_access()  # Body reads can span an invitation's deadline.
        # Cloudflare sets this header; it is used only for a coarse abuse bound,
        # never authentication or persisted identity. Queue bounds apply globally.
        ip = self.headers.get('CF-Connecting-IP', self.client_address[0])[:100]
        app.rates.check('ip:'+hashlib.sha256(ip.encode()).hexdigest(), 240)
        if path == '/api/session':
            self._fields(body, ())
            token, conversation = app.store.bootstrap(self._token())
            cookie = f'seminar_session={token}; Path={access["prefix"]}/; HttpOnly; SameSite=Lax; Max-Age=86400'+('; Secure' if access['secure'] else '')
            return self._json(200, app.session_response(token, conversation, access), cookie)
        token = self._authenticate()
        if path == '/api/new':
            self._fields(body, ())
            app.rates.check('new:'+app.store.digest(token), 12)
            return self._json(200, app.session_response(token, app.store.new(token), access))
        if path == '/api/chat':
            return self._chat(token, body)
        if path == '/api/classify':
            return self._classify(token, body)
        self._fields(body, ('nickname','comment','consent','expectedConversationId','expectedMessageCount'))
        if body['consent'] is not True:
            raise SeminarError(400, 'consent_required', 'Choose to share this conversation before submitting.')
        nickname = self._text(body['nickname'], 40, True)
        comment = self._text(body['comment'], 500, True)
        expected_id, expected_count = body['expectedConversationId'], body['expectedMessageCount']
        if not isinstance(expected_id, str) or len(expected_id) != 32 or type(expected_count) is not int or not 2 <= expected_count <= 20 or expected_count % 2:
            raise SeminarError(400, 'invalid_input', 'The displayed conversation version is required.')
        with self._access_guard():
            submission = app.store.submit(token, nickname, comment, app.model.metadata(), expected_id, expected_count)
        return self._json(200, submission)

    def _handle(self):
        try:
            self._run()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception as exc:
            error = exc if isinstance(exc, SeminarError) else SeminarError(503, 'unavailable', 'The demo is temporarily unavailable. Reload to check your conversation before retrying.')
            if not isinstance(exc, SeminarError):
                print('Seminar request failed: '+type(exc).__name__, flush=True)
            try:
                if self.streaming:
                    self._event({'type':'error', 'code':error.code, 'message':error.message})
                elif error.code == 'share_closed' and self.command in ('GET', 'HEAD') and 'text/html' in self.headers.get('Accept', ''):
                    body = b'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Invitation closed</title><body><h1>This seminar invitation has closed.</h1><p>Ask the presenter for a new QR code or link. Submitted conversations remain available for discussion.</p></body></html>'
                    self._headers(410, 'text/html; charset=utf-8', len(body))
                    if self.command != 'HEAD':
                        self.wfile.write(body)
                else:
                    self._json(error.status, {'error':{'code':error.code, 'message':error.message}})
            except OSError:
                pass

    do_GET = do_HEAD = do_POST = _handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-bind', default='127.0.0.1')
    parser.add_argument('--public-port', type=int, default=18870)
    parser.add_argument('--admin-port', type=int, default=18871)
    parser.add_argument('--public-url')
    parser.add_argument('--share-state')
    parser.add_argument('--tunnel-endpoint')
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--static', required=True)
    parser.add_argument('--model-updates')
    parser.add_argument('--training-export-url')
    args = parser.parse_args()
    if bool(args.share_state) != bool(args.tunnel_endpoint) or bool(args.public_url) == bool(args.share_state):
        parser.error('Choose a static public URL or both sharing state and tunnel endpoint paths.')
    from seminar_publish import EXPORT_URL, SeminarPublisher
    if args.training_export_url not in (None, EXPORT_URL):
        parser.error('The training export URL must be the fixed local training bridge route.')
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    share = None
    if args.share_state:
        from seminar_share import ShareManager
        share = ShareManager(args.share_state, args.tunnel_endpoint)
    publisher = SeminarPublisher(args.model_updates or Path(args.state).parent / 'models',
                                 args.bundle, args.training_export_url)
    app = AudienceApplication(SeminarStore(args.state), publisher.current_model, args.public_url,
                              args.static, share=share, publisher=publisher)
    public = AudienceHTTPServer((args.public_bind, args.public_port), app)
    private = AudienceHTTPServer(('127.0.0.1', args.admin_port), app, private=True)
    threading.Thread(target=private.serve_forever, daemon=True).start()
    print('Seminar audience ready; snapshot '+str(app.model.metadata().get('snapshotId')), flush=True)
    try:
        public.serve_forever()
    finally:
        public.server_close()
        private.shutdown()
        private.server_close()


if __name__ == '__main__':
    main()
