import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

spec = importlib.util.spec_from_file_location("local_server", Path(__file__).resolve().parents[1] / "local_server.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


def native_completion():
    return {"content": " animal", "generation_settings": {"post_sampling_probs": False},
            "stop_type": "limit", "timings": {"predicted_n": 1}, "completion_probabilities": [
                {"id": 9572, "token": " animal", "bytes": list(b" animal"), "logprob": math.log(.4),
                 "top_logprobs": [{"id": 9572, "token": " animal", "logprob": math.log(.4)},
                                  {"id": 9209, "token": " independent", "logprob": math.log(.2)}]}]}


class ProbabilityTests(unittest.TestCase):
    def test_top_k_mass_and_sample_probability_are_not_renormalized(self):
        result = gateway.normalize_completion(native_completion())
        self.assertAlmostEqual(result["tokens"][0]["p"], .4)
        self.assertAlmostEqual(sum(x["p"] for x in result["tokens"][0]["candidates"]), .6)
        self.assertFalse(result["finished"])
        self.assertEqual(result["probabilityKind"], "pre-sampling")

    def test_eos_remains_a_real_token_even_with_empty_display_piece(self):
        native = native_completion()
        native["stop_type"] = "eos"
        native["completion_probabilities"][0].update(id=248046, token="", bytes=[])
        result = gateway.normalize_completion(native)
        self.assertTrue(result["finished"])
        self.assertEqual(result["tokens"][0]["id"], 248046)
        self.assertEqual(result["tokens"][0]["bytes"], [])

    def test_unknown_probability_semantics_and_invalid_numbers_fail(self):
        for invalid in (True, None, "false"):
            native = native_completion()
            native["generation_settings"]["post_sampling_probs"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(gateway.ApiError):
                gateway.normalize_completion(native)
        for invalid in (float("nan"), float("inf"), .1, True, None):
            native = native_completion()
            native["completion_probabilities"][0]["logprob"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(gateway.ApiError):
                gateway.normalize_completion(native)

    def test_broken_probability_distributions_and_missing_records_fail(self):
        native = native_completion()
        native["completion_probabilities"][0]["top_logprobs"][1]["logprob"] = math.log(.9)
        with self.assertRaises(gateway.ApiError): gateway.normalize_completion(native)
        native = native_completion()
        native["completion_probabilities"] = []
        with self.assertRaises(gateway.ApiError): gateway.normalize_completion(native)


class InputTests(unittest.TestCase):
    def test_generation_boundaries(self):
        valid = {"tokens": [32, 7993, 369, 449], "steps": 1, "temperature": 0, "seed": 42}
        self.assertEqual(gateway.validate_generate(valid, 248320), valid)
        for key, value in (("tokens", []), ("tokens", [True]), ("tokens", [248320]),
                           ("tokens", [32] * 2049), ("steps", 33), ("steps", True),
                           ("temperature", float("nan")), ("temperature", True),
                           ("temperature", 1.51), ("seed", -1)):
            invalid = dict(valid, **{key: value})
            with self.subTest(key=key, value=str(value)[:30]), self.assertRaises(gateway.ApiError):
                gateway.validate_generate(invalid, 248320)
        with self.assertRaises(gateway.ApiError): gateway.validate_generate(dict(valid, url="https://example.com"))

    def test_prepare_input_boundaries(self):
        gateway.validate_prepare({"text": "A cat is an", "mode": "raw"})
        for body in ({"text": "", "mode": "raw"}, {"text": "x" * 8193, "mode": "raw"},
                     {"text": "hi", "mode": "chat"}, {"text": "hi"}, []):
            with self.assertRaises(gateway.ApiError): gateway.validate_prepare(body)

    def test_generation_forwards_exact_ids_and_sampler_controls(self):
        upstream = gateway.Upstream(1)
        calls = []
        upstream.model = lambda: {"meta": {"n_vocab": 248320}}
        upstream.request = lambda path, body: (calls.append((path, body)) or native_completion())
        upstream.generate({"tokens": [32, 7993, 369, 449], "steps": 1, "temperature": .7, "seed": 42})
        self.assertEqual(calls[0][1]["prompt"], [32, 7993, 369, 449])
        self.assertEqual((calls[0][1]["top_k"], calls[0][1]["top_p"], calls[0][1]["min_p"]), (0, 1, 0))
        self.assertFalse(calls[0][1]["stream"])

    def test_instruction_uses_actual_template_and_special_token_parsing(self):
        upstream = gateway.Upstream(1)
        calls = []
        def request(path, body):
            calls.append((path, body))
            if path == "/apply-template": return {"prompt": "<|im_start|>user\nHi"}
            return {"tokens": [{"id": 248045, "piece": "<|im_start|>"}]}
        upstream.request = request
        result = upstream.prepare({"text": "Hi", "mode": "instruction"})
        self.assertFalse(calls[0][1]["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(calls[1][1]["parse_special"])
        self.assertEqual(result["tokens"][0]["id"], 248045)

    def test_generation_is_serialized_and_lock_released_on_failure(self):
        upstream = gateway.Upstream(1)
        body = {"tokens": [32], "steps": 1, "temperature": 0, "seed": 42}
        upstream.generation_lock.acquire()
        with self.assertRaises(gateway.ApiError) as error: upstream.generate(body)
        self.assertEqual(error.exception.status, 503)
        upstream.generation_lock.release()
        def failed(): raise gateway.ApiError(503, "offline", "offline")
        upstream.model = failed
        with self.assertRaises(gateway.ApiError): upstream.generate(body)
        self.assertTrue(upstream.generation_lock.acquire(blocking=False))
        upstream.generation_lock.release()

    def test_split_utf8_input_tokens_keep_exact_ids_and_display_bytes(self):
        upstream = gateway.Upstream(1)
        upstream.request = lambda *args: {"tokens": [{"id": 10838, "piece": [32, 240, 159]}, {"id": 234, "piece": [140]}]}
        result = upstream.prepare({"text": " 🌍", "mode": "raw"})
        self.assertEqual(result["tokens"][0], {"id": 10838, "piece": "⟨20 F0 9F⟩", "bytes": [32, 240, 159]})
        self.assertEqual(result["tokens"][1]["id"], 234)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = gateway.GatewayServer(("127.0.0.1", 0), gateway.GatewayHandler)
        cls.server.upstream = gateway.Upstream(1)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, data=None, **headers):
        return self.opener.open(urllib.request.Request(self.url + path, data=data, headers=headers), timeout=2)

    def test_health_identifies_application_without_claiming_model_online(self):
        with self.request("/api/health") as response:
            body = json.load(response)
        self.assertEqual(body, {"app": gateway.APP_ID, "status": "ok"})

    def test_training_pages_csp_permits_only_fixed_loopback_runtime_connections(self):
        # The UI is served on8768 but GPU training/telemetry live on8769. A
        # self-only connect-src silently blocks both before CORS can run.
        expected = {"'self'", "http://127.0.0.1:8769", "http://localhost:8769"}
        for path in ("/", "/training.html"):
            with self.subTest(path=path), self.request(path) as response:
                policies = response.headers.get_all("Content-Security-Policy", [])
                self.assertEqual(len(policies), 1)
                directives = {}
                for part in policies[0].split(";"):
                    tokens = part.split()
                    if tokens:
                        self.assertNotIn(tokens[0], directives)
                        directives[tokens[0]] = tokens[1:]
                connections = directives.get("connect-src", [])
                self.assertEqual(set(connections), expected)
                self.assertFalse(any("*" in source for source in connections))
                self.assertNotIn("http:", connections)
                self.assertNotIn("https:", connections)
                self.assertIn(b"training-app.js", response.read())

    def test_foreign_origin_and_host_are_rejected_before_model_access(self):
        for headers in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers), self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/health", **headers)
            self.assertEqual(error.exception.code, 403)

    def test_malformed_json_and_oversized_body_are_rejected(self):
        for body, code in ((b"{broken", 400), (b"x" * 65537, 413)):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/prepare", data=body, **{"Content-Type": "application/json"})
            self.assertEqual(error.exception.code, code)
            self.assertIn("error", json.load(error.exception))


class TunnelTests(unittest.TestCase):
    def test_reconnect_only_when_owned_process_exited(self):
        tunnel = gateway.Tunnel(18091)
        tunnel.process = Mock()
        tunnel.process.poll.return_value = None
        tunnel._start = Mock()
        tunnel.ensure_running()
        tunnel._start.assert_not_called()
        tunnel.process.poll.return_value = 255
        tunnel.ensure_running()
        tunnel._start.assert_called_once()
        tunnel.stop()
        with self.assertRaises(RuntimeError): tunnel.ensure_running()

    def test_connection_failure_is_exposed_and_completion_is_not_replayed(self):
        def unavailable(): raise RuntimeError("ssh unavailable")
        upstream = gateway.Upstream(1, unavailable)
        upstream.request = Mock()
        with self.assertRaises(gateway.ApiError) as error:
            upstream.generate({"tokens": [32], "steps": 1, "temperature": 0, "seed": 42})
        self.assertEqual(error.exception.status, 503)
        upstream.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class ConfiguredModelTunnelTests(unittest.TestCase):
    def test_missing_or_invalid_destination_never_opens_a_socket_or_spawns_ssh(self):
        for value in ('','  ','-oProxyCommand=bad','host name','host\nname'):
            with self.subTest(value=value), patch.dict(os.environ,{'GPT_MODEL_SSH_HOST':value}), patch.object(gateway.socket,'socket') as socket, patch.object(gateway.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError,'GPT_MODEL_SSH_HOST'):
                    gateway.Tunnel(18091).start()
                socket.assert_not_called();spawn.assert_not_called()

    def test_configured_model_destination_is_used_as_one_ssh_argument(self):
        with patch.dict(os.environ,{'GPT_MODEL_SSH_HOST':'test-user@model.example.com'}), patch.object(gateway.shutil,'which',return_value='ssh'), patch.object(gateway.socket,'socket') as socket, patch.object(gateway.subprocess,'Popen') as spawn:
            socket.return_value.__enter__.return_value.connect_ex.side_effect=[1,0]
            spawn.return_value.poll.return_value=None
            tunnel=gateway.Tunnel(18091);tunnel.start()
            self.assertEqual(spawn.call_args.args[0][-1],'test-user@model.example.com')
            self.assertEqual(spawn.call_count,1);tunnel.stop()
