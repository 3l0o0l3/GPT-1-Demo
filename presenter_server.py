"""Authenticated hosted presenter; fixed proxies, no model or SMTP secrets in UI."""
import argparse
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit

from presenter_auth import AuthError, AuthService
from presenter_mail import GmailMailer
from runtime_service import GpuUpstream, RuntimeHandler, RuntimeFailure
from seminar_gateway import SeminarGateway, SeminarGatewayError

LOGIN_COOKIE = '__Host-presenter_login'
SESSION_COOKIE = '__Host-presenter_session'
ASSETS = frozenset(('index.html','training.html','training-app.js','training-jobs.js','training-style.css',
    'local-style.css','telemetry.js','telemetry.css','seminar-panel.js','seminar-panel.css',
    'token-stream.js','token-playback.js','presenter-session.js','teaching.html','style.css',
    'app.js','engine.js','worker.js','vendor/katex.min.js','vendor/katex.min.css'))
LOGIN_ASSETS = {'/login':'presenter-login.html','/presenter-login.js':'presenter-login.js',
                '/presenter-login.css':'presenter-login.css'}


class FixedConnection:
    """The dedicated reverse SSH service owns transport availability/lifecycle."""
    def __init__(self, port):
        self.port = port

    def ensure_running(self):
        pass


class PresenterApplication:
    def __init__(self, auth, mailer, static_dir, public_url):
        parsed = urlsplit(public_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
            raise ValueError('Presenter public URL must be a single HTTPS origin.')
        self.auth, self.mailer = auth, mailer
        self.origin = 'https://'+parsed.netloc
        self.host = parsed.netloc
        self.static_dir = Path(static_dir).resolve(strict=True)
        self.assets = set(ASSETS)
        for path in (self.static_dir/'vendor'/'fonts').glob('KaTeX_*'):
            if re.fullmatch(r'KaTeX_[A-Za-z0-9]+-[A-Za-z]+\.(?:woff2?|ttf)', path.name):
                self.assets.add('vendor/fonts/'+path.name)


class PresenterHandler(RuntimeHandler):
    server_version = 'GPTPresenter/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def _headers(self, status, kind, length=None, cookies=()):
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header('Strict-Transport-Security','max-age=86400')
        if length is not None:
            self.send_header('Content-Length',str(length))
        for cookie in cookies:
            self.send_header('Set-Cookie',cookie)
        self.send_header('Connection','close')
        self.close_connection = True
        self.end_headers()

    def _json(self, status, payload, cookies=()):
        body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
        self._headers(status,'application/json; charset=utf-8',len(body),cookies)
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _stream_headers(self):
        self._headers(200,'application/x-ndjson; charset=utf-8')

    @staticmethod
    def _cookie_header(name, value, seconds):
        return f'{name}={value}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={seconds}'

    def _cookie(self, name):
        try:
            cookies=SimpleCookie(self.headers.get('Cookie',''))
            value=cookies[name].value if name in cookies else None
            return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{43}',value) else None
        except Exception:
            return None

    def _boundary(self):
        app=self.server.app
        hosts=self.headers.get_all('Host',[])
        origins=self.headers.get_all('Origin',[])
        if len(hosts)!=1 or hosts[0]!=app.host:
            raise AuthError(403,'invalid_host','Use the configured presenter URL.')
        if origins and (len(origins)!=1 or origins[0]!=app.origin):
            raise AuthError(403,'invalid_origin','Use the presenter page to send requests.')
        if self.command=='POST' and (origins!=[app.origin] or self.headers.get('Sec-Fetch-Site')=='cross-site'):
            raise AuthError(403,'invalid_origin','Use the presenter page to send requests.')

    def _body(self):
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1 or self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
            raise AuthError(400,'invalid_input','Send one bounded JSON object.')
        try:
            length=int(self.headers['Content-Length'])
            if not 0<length<=65536:
                raise AuthError(413,'body_limit','The request exceeds the size limit.')
            raw=self.rfile.read(length)
            if len(raw)!=length:
                raise ValueError()
            def unique(pairs):
                value=dict(pairs)
                if len(value)!=len(pairs):
                    raise ValueError()
                return value
            value=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
            if not isinstance(value,dict):
                raise ValueError()
            return value
        except (ValueError,UnicodeError,RecursionError):
            raise AuthError(400,'invalid_input','Send a valid JSON object.')

    @staticmethod
    def _fields(body, names):
        if set(body)!=set(names):
            raise AuthError(400,'invalid_input','Unexpected or missing request fields.')

    def _redirect(self, location):
        self.send_response(302)
        self.send_header('Location',location)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length','0')
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection=True

    def _asset(self,name):
        path=self.server.app.static_dir/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(self.server.app.static_dir):
            raise AuthError(404,'not_found','Not found.')
        data=path.read_bytes()
        kind=mimetypes.guess_type(name)[0] or 'application/octet-stream'
        if name.endswith('.js'):
            kind='text/javascript'
        self._headers(200,kind,len(data))
        if self.command!='HEAD':
            self.wfile.write(data)

    def _session(self):
        app=self.server.app
        token=self._cookie(SESSION_COOKIE)
        if token:
            try:
                return self._json(200,dict(authenticated=True,**app.auth.require(token)))
            except AuthError as error:
                if error.status!=401:
                    raise
        login=self._cookie(LOGIN_COOKIE) or secrets.token_urlsafe(32)
        return self._json(200,dict(authenticated=False,csrfToken=app.auth.csrf(login),smtpConfigured=app.mailer.configured),
                          [self._cookie_header(LOGIN_COOKIE,login,3600)])

    def _login_action(self,path,body):
        app=self.server.app
        browser=self._cookie(LOGIN_COOKIE)
        supplied=self.headers.get('X-CSRF-Token','')
        if not browser or not re.fullmatch(r'[a-f0-9]{64}',supplied) or not secrets.compare_digest(supplied,app.auth.csrf(browser)):
            raise AuthError(403,'csrf','Reload the login page before trying again.')
        if path=='/api/auth/request-code':
            self._fields(body,('email',))
            # This identifier is an additional abuse bound, never an identity assertion.
            client=self.headers.get('CF-Connecting-IP',self.client_address[0])[:100]
            return self._json(200,app.auth.issue(body['email'],browser,client))
        self._fields(body,('challengeId','code'))
        result=app.auth.verify(body['challengeId'],body['code'],browser)
        token=result['sessionToken']
        public={k:v for k,v in result.items() if k!='sessionToken'}
        return self._json(200,dict(authenticated=True,**public),
                          [self._cookie_header(SESSION_COOKIE,token,28800),self._cookie_header(LOGIN_COOKIE,'',0)])

    def _run(self):
        self._boundary()
        app=self.server.app
        parsed=urlsplit(self.path)
        path=parsed.path
        if parsed.scheme or parsed.netloc or parsed.fragment or '\\' in path:
            raise AuthError(400,'invalid_path','Invalid request path.')
        if self.command in ('GET','HEAD'):
            if path in LOGIN_ASSETS and (not parsed.query or path=='/login'):
                return self._asset(LOGIN_ASSETS[path])
            if path=='/api/auth/session' and not parsed.query:
                return self._session()
        token=self._cookie(SESSION_COOKIE)
        if self.command=='POST' and path in ('/api/auth/request-code','/api/auth/verify') and not parsed.query:
            return self._login_action(path,self._body())
        try:
            app.auth.require(token)
        except AuthError as error:
            if error.status==401 and self.command in ('GET','HEAD') and path in ('/','/index.html','/training.html','/teaching.html'):
                return self._redirect('/login')
            raise
        body=None
        if self.command=='POST':
            app.auth.require(token,csrf=self.headers.get('X-CSRF-Token',''))
            body=self._body()
            # Reject an expired/revoked session after a possibly slow body upload.
            app.auth.require(token,csrf=self.headers.get('X-CSRF-Token',''))
            if path=='/api/auth/logout' and not parsed.query:
                self._fields(body,())
                app.auth.logout(token)
                return self._json(200,{'authenticated':False},[self._cookie_header(SESSION_COOKIE,'',0)])
        if path.startswith('/api/training/') and self.command in ('GET','POST'):
            return self._forward(self.command,json.dumps(body,allow_nan=False).encode() if body is not None else None)
        if self.command=='GET' and path=='/api/telemetry' and not parsed.query:
            status,value=self.server.gpu.request('GET','/api/telemetry',timeout=8)
            return self._json(status,value)
        if path.startswith('/api/seminar/'):
            if self.command=='GET':
                return self._json(200,self.server.seminar.get(self.path))
            if self.command=='POST':
                return self._json(200,self.server.seminar.post(self.path,body))
        if self.command in ('GET','HEAD') and not parsed.query:
            name='index.html' if path=='/' else path.removeprefix('/')
            if name in app.assets:
                return self._asset(name)
        raise AuthError(404,'not_found','Not found.')

    def _handle(self):
        try:
            self._run()
        except (BrokenPipeError,ConnectionResetError,TimeoutError):
            pass
        except Exception as exc:
            if isinstance(exc,(AuthError,SeminarGatewayError)):
                status,code,message=exc.status,exc.code,exc.message
            elif isinstance(exc,RuntimeFailure):
                status,code,message=503,'gpu_unavailable',str(exc)
            else:
                status,code,message=503,'unavailable','The presenter service is temporarily unavailable.'
                print('Presenter request failed: '+type(exc).__name__,flush=True)
            try:
                self._json(status,{'error':{'code':code,'message':message}})
            except OSError:
                pass

    do_GET=do_HEAD=do_POST=_handle

    def do_OPTIONS(self):
        self._json(403,{'error':{'code':'forbidden','message':'Cross-origin access is unsupported.'}})


class PresenterHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    request_queue_size=64

    def __init__(self,address,app,gpu,seminar):
        self.app,self.gpu,self.seminar=app,gpu,seminar
        self.connections=threading.BoundedSemaphore(64)
        super().__init__(address,PresenterHandler)

    def process_request(self,request,client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request,client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self,request,client_address):
        try:
            super().process_request_thread(request,client_address)
        finally:
            self.connections.release()


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--bind',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=18880)
    parser.add_argument('--public-url',required=True)
    parser.add_argument('--allowed-email',action='append',required=True,help='Allowed sign-in recipient; repeat for each presenter.')
    parser.add_argument('--static',required=True)
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--smtp-config',required=True)
    parser.add_argument('--training-port',type=int,default=18769)
    args=parser.parse_args(argv)
    state=Path(args.state_dir);state.mkdir(parents=True,exist_ok=True,mode=0o700)
    mailer=GmailMailer(args.smtp_config,allowed_email=args.allowed_email)
    auth=AuthService(state/'auth.sqlite3',state/'auth.key',mailer,allowed_email=args.allowed_email)
    app=PresenterApplication(auth,mailer,args.static,args.public_url)
    server=PresenterHTTPServer((args.bind,args.port),app,
        GpuUpstream(FixedConnection(args.training_port),port=args.training_port),
        SeminarGateway(FixedConnection(18871)))
    print('Authenticated presenter ready.',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__=='__main__':
    main()
