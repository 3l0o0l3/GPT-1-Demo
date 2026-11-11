"""Loopback-only teaching gateway to the configured remote llama.cpp service over SSH.

No model is downloaded or changed. Run with Python 3.10+; standard library only.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from training_service import TrainingService, TrainingApiError
from seminar_gateway import SeminarGateway, SeminarGatewayError

APP_ID = "local-qwen-learning-lab-v1"
ROOT = Path(__file__).resolve().parent
MAX_BODY = 65536


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_prepare(body):
    if not isinstance(body, dict) or set(body) != {"text", "mode"}:
        raise ApiError(400, "invalid_input", "Provide exactly text and mode.")
    if not isinstance(body["text"], str) or not 1 <= len(body["text"]) <= 8192:
        raise ApiError(400, "invalid_input", "Text must contain 1–8192 characters.")
    if body["mode"] not in ("raw", "instruction"):
        raise ApiError(400, "invalid_input", "Mode must be raw or instruction.")
    return body


def validate_generate(body, vocab_size=None):
    if not isinstance(body, dict) or set(body) != {"tokens", "steps", "temperature", "seed"}:
        raise ApiError(400, "invalid_input", "Provide exactly tokens, steps, temperature and seed.")
    ids = body["tokens"]
    ceiling = vocab_size if vocab_size is not None else 2147483647
    if not isinstance(ids, list) or not 1 <= len(ids) <= 2048 or any(
        not is_int(token) or not 0 <= token < ceiling for token in ids
    ):
        raise ApiError(400, "invalid_input", "Provide 1–2048 valid integer token IDs.")
    if not is_int(body["steps"]) or not 1 <= body["steps"] <= 32:
        raise ApiError(400, "invalid_input", "Steps must be an integer from 1 to 32.")
    temp = body["temperature"]
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(temp) or not 0 <= temp <= 1.5:
        raise ApiError(400, "invalid_input", "Temperature must be between 0 and 1.5.")
    if not is_int(body["seed"]) or not 0 <= body["seed"] <= 2147483647:
        raise ApiError(400, "invalid_input", "Seed must be an integer from 0 to 2147483647.")
    return body


def malformed(message="The model returned an unsupported response."):
    return ApiError(502, "invalid_upstream_response", message)


def probability_record(record):
    if not isinstance(record, dict) or not is_int(record.get("id")) or record["id"] < 0 or not isinstance(record.get("token"), str):
        raise malformed("The model did not return valid token IDs and pieces.")
    logprob = record.get("logprob")
    if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob) or logprob > 0:
        raise malformed("The model did not return finite token log probabilities.")
    return {"id": record["id"], "piece": record["token"], "logprob": logprob, "p": math.exp(logprob)}


def normalize_completion(result):
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise malformed()
    settings = result.get("generation_settings")
    if not isinstance(settings, dict) or settings.get("post_sampling_probs") is not False:
        raise malformed("This server has not confirmed pre-sampling probability semantics.")
    rows = result.get("completion_probabilities")
    if not isinstance(rows, list) or not rows or len(rows) > 32:
        raise malformed("The model did not return token-level probability records.")
    tokens = []
    for row in rows:
        record = probability_record(row)
        raw_bytes = row.get("bytes")
        candidates = row.get("top_logprobs")
        if not isinstance(raw_bytes, list) or any(not is_int(b) or not 0 <= b <= 255 for b in raw_bytes):
            raise malformed("The model did not return valid token bytes.")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
            raise malformed("The model did not return candidate probabilities.")
        normalized = [probability_record(candidate) for candidate in candidates]
        if len({c["id"] for c in normalized}) != len(normalized) or sum(c["p"] for c in normalized) > 1.0001:
            raise malformed("The model returned an invalid candidate probability distribution.")
        record.update(bytes=raw_bytes, candidates=normalized)
        tokens.append(record)
    stop = result.get("stop_type")
    if stop not in ("limit", "eos", "word"):
        raise malformed("The model returned an unknown stopping reason.")
    timings = result.get("timings")
    if not isinstance(timings, dict):
        raise malformed("The model did not return timing information.")
    return {"content": result["content"], "tokens": tokens, "stopType": stop,
            "finished": stop != "limit", "timings": timings, "probabilityKind": "pre-sampling"}


class Upstream:
    def __init__(self, port, ensure_connection=None):
        self.base = f"http://127.0.0.1:{port}"
        # Never route loopback traffic through a user's HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.generation_lock = threading.Lock()
        self.ensure_connection = ensure_connection

    def connect(self):
        if self.ensure_connection:
            try:
                self.ensure_connection()
            except (OSError, RuntimeError) as error:
                raise ApiError(503, "ssh_unavailable", "Cannot reconnect to the configured SSH model host. Check the connection, then use Retry connection.") from error

    def request(self, path, body=None, timeout=60):
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise malformed("The model response exceeded the size limit.")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise malformed()
            return result
        except urllib.error.HTTPError as error:
            if error.code in (429, 503):
                raise ApiError(503, "model_busy", "The model is busy. Try again when the current request finishes.") from error
            raise ApiError(502, "upstream_error", f"The model service returned HTTP {error.code}.") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as error:
            raise ApiError(503, "model_unavailable", "Cannot reach the configured remote model. Check its SSH host and model service, then retry.") from error
        except (ValueError, UnicodeError) as error:
            raise malformed("The model did not return valid JSON.") from error

    def model(self):
        self.connect()
        self.request("/health", timeout=8)
        result = self.request("/v1/models", timeout=8)
        data = result.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not isinstance(data[0].get("id"), str):
            raise malformed("The model service did not identify its loaded model.")
        meta = data[0].get("meta", {})
        if not isinstance(meta, dict):
            raise malformed()
        return {"model": data[0]["id"], "online": True, "meta": meta,
                "capabilities": {"tokenize": True, "logprobs": True, "training": False, "attentions": False}}

    def prepare(self, body):
        validate_prepare(body)
        self.connect()
        prompt = body["text"]
        if body["mode"] == "instruction":
            result = self.request("/apply-template", {"messages": [{"role": "user", "content": prompt}],
                                  "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
            prompt = result.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise malformed("The model service did not return a chat template.")
        result = self.request("/tokenize", {"content": prompt, "with_pieces": True,
                              "add_special": False, "parse_special": True})
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise malformed("The model did not return tokenization.")
        if len(tokens) > 2048:
            raise ApiError(400, "context_limit", "This demo supports at most 2048 prompt tokens. Shorten the prompt.")
        normalized = []
        for token in tokens:
            if not isinstance(token, dict) or not is_int(token.get("id")) or token["id"] < 0:
                raise malformed("The model returned invalid tokenization.")
            piece = token.get("piece")
            if isinstance(piece, str):
                normalized.append({"id": token["id"], "piece": piece})
            elif isinstance(piece, list) and piece and all(is_int(b) and 0 <= b <= 255 for b in piece):
                # A token may contain only part of a UTF-8 character. Display exact
                # byte notation instead of inventing replacement characters.
                normalized.append({"id": token["id"], "piece": "⟨" + " ".join(f"{b:02X}" for b in piece) + "⟩", "bytes": piece})
            else:
                raise malformed("The model returned invalid token pieces.")
        return {"mode": body["mode"], "formattedPrompt": prompt, "tokens": normalized}

    def generate(self, body):
        validate_generate(body)
        if not self.generation_lock.acquire(blocking=False):
            raise ApiError(503, "model_busy", "Another generation is running in this demo. Wait for it to finish.")
        try:
            meta = self.model()["meta"]
            vocab = meta.get("n_vocab")
            if not is_int(vocab) or vocab <= 0:
                raise malformed("The model service did not return its vocabulary size.")
            validate_generate(body, vocab)
            result = self.request("/completion", {"prompt": body["tokens"], "n_predict": body["steps"],
                                  "temperature": body["temperature"], "seed": body["seed"],
                                  "n_probs": 8, "top_k": 0, "top_p": 1, "min_p": 0,
                                  "cache_prompt": True, "stream": False, "post_sampling_probs": False})
            return normalize_completion(result)
        finally:
            self.generation_lock.release()


class GatewayHandler(SimpleHTTPRequestHandler):
    server_version = "LocalQwenDemo/1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' http://127.0.0.1:8769 http://localhost:8769; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()

    def check_local_request(self):
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in allowed_hosts:
            raise ApiError(403, "forbidden_host", "Use the local demo URL on 127.0.0.1.")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed_hosts}:
            raise ApiError(403, "forbidden_origin", "This local API only accepts requests from its own page.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise ApiError(403, "forbidden_origin", "Cross-site requests are not supported.")

    def send_json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def api_error(self, error):
        self.send_json(error.status, {"error": {"code": error.code, "message": error.message}})

    def do_GET(self):
        try:
            self.check_local_request()
            if self.path.startswith('/api/seminar/'):
                return self.send_json(200, self.server.seminar.get(self.path))
            if self.path.startswith('/api/training/'):
                return self.send_json(200, self.server.training.dispatch('GET', self.path))
            if self.path == "/api/health":
                return self.send_json(200, {"app": APP_ID, "status": "ok"})
            if self.path == "/api/model":
                return self.send_json(200, self.server.upstream.model())
            if self.path.startswith("/api/"):
                raise ApiError(404, "not_found", "Unknown API route.")
            super().do_GET()
        except (ApiError, TrainingApiError, SeminarGatewayError) as error:
            self.api_error(error)
        except Exception as error:
            print(f'Gateway GET error: {type(error).__name__}', file=sys.stderr, flush=True)
            self.api_error(ApiError(500, 'internal_error', 'The local training service could not complete this request.'))

    def do_HEAD(self):
        try:
            self.check_local_request()
            super().do_HEAD()
        except ApiError as error:
            self.api_error(error)

    def do_POST(self):
        try:
            self.check_local_request()
            if self.path not in ("/api/prepare", "/api/generate", "/api/seminar/share", "/api/seminar/share/revoke", "/api/seminar/submissions/clear", "/api/seminar/model/sync") and not self.path.startswith('/api/training/'):
                raise ApiError(404, "not_found", "Unknown API route.")
            if self.headers.get_content_type() != "application/json":
                raise ApiError(400, "invalid_input", "Send application/json.")
            length = self.headers.get("Content-Length", "")
            if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1 or not length.isdigit():
                raise ApiError(400, "invalid_input", "Content-Length is required.")
            length = int(length)
            if length > MAX_BODY:
                raise ApiError(413, "body_too_large", "Request body exceeds 64 KiB.")
            try:
                raw = self.rfile.read(length)
                body = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ApiError(400, "invalid_input", "Request body must be valid JSON.") from error
            if self.path.startswith('/api/seminar/'):
                result = self.server.seminar.post(self.path, body)
            elif self.path.startswith('/api/training/'):
                result = self.server.training.dispatch('POST', self.path, body)
            else:
                result = self.server.upstream.prepare(body) if self.path == "/api/prepare" else self.server.upstream.generate(body)
            self.send_json(200, result)
        except (ApiError, TrainingApiError, SeminarGatewayError) as error:
            self.api_error(error)
        except Exception as error:
            # Do not include request text or private service details in logs/errors.
            print(f"Gateway internal error: {type(error).__name__}", file=sys.stderr, flush=True)
            self.api_error(ApiError(500, "internal_error", "The local gateway could not complete this request."))

    def list_directory(self, path):
        self.send_error(404)
        return None

    def log_message(self, fmt, *args):
        # Paths contain no user prompts; bodies are deliberately never logged.
        print(f"{self.log_date_time_string()} {fmt % args}", file=sys.stderr, flush=True)


class Tunnel:
    def __init__(self, port):
        self.port = port
        self.process = None
        self.lock = threading.Lock()
        self.closing = False
        atexit.register(self.stop)

    def start(self):
        with self.lock:
            self._start()

    def ensure_running(self):
        # Reconnect only before a new operation, never replay a completion.
        with self.lock:
            if self.closing:
                raise RuntimeError("The local gateway is shutting down.")
            if self.process is None or self.process.poll() is not None:
                self._start()

    def _start(self):
        host = os.environ.get('GPT_MODEL_SSH_HOST', '').strip()
        if not host:
            raise RuntimeError('Set GPT_MODEL_SSH_HOST before connecting the optional remote model.')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]{0,254}', host):
            raise RuntimeError('GPT_MODEL_SSH_HOST must be a single SSH destination, without options or whitespace.')
        ssh = shutil.which("ssh")
        if not ssh:
            raise RuntimeError("OpenSSH is missing. Install the Windows OpenSSH client.")
        # Refuse to reuse an unowned listener accidentally pointing at another model.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"Local tunnel port {self.port} is already in use. Close the previous demo instance or choose --tunnel-port.")
        command = [ssh, "-N", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                   "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                   "-o", "ServerAliveCountMax=2", "-L", f"127.0.0.1:{self.port}:127.0.0.1:18089", host]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("SSH could not connect to the configured model host.")
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.1)
        self._stop_process()
        raise RuntimeError("SSH tunnel did not become ready within 15 seconds.")

    def stop(self):
        with self.lock:
            self.closing = True
            self._stop_process()

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(70)
        return connection, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--tunnel-port", type=int, default=18091)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1024 <= args.tunnel_port <= 65535:
        parser.error("Ports must be between 1024 and 65535.")
    tunnel = Tunnel(args.tunnel_port)
    server = None
    try:
        # Bind the page port first so a duplicate launch cannot create a spare tunnel.
        server = GatewayServer(("127.0.0.1", args.port), GatewayHandler)
        server.upstream = Upstream(args.tunnel_port, tunnel.ensure_running)
        server.training = TrainingService(ROOT)
        server.seminar = SeminarGateway()
        try:
            tunnel.start()
        except (OSError, RuntimeError):
            print('The optional SSH model is unavailable; local training remains available.', file=sys.stderr, flush=True)
        print(f"{APP_ID} ready at http://127.0.0.1:{args.port}", flush=True)
        def request_stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, request_stop)
        server.serve_forever(poll_interval=0.25)
        return 0
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        if server:
            server.server_close()
            if hasattr(server, 'seminar'):
                server.seminar.tunnel.stop()
        tunnel.stop()


if __name__ == "__main__":
    raise SystemExit(main())
