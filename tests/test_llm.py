import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from usecase_gen.llm import LLMError, OpenAICompatClient


def _server(statuses: list[int]):
    """Serves the given HTTP statuses in order, then 200 with a chat reply."""
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            code = statuses[len(calls)] if len(calls) < len(statuses) else 200
            calls.append(code)
            body = json.dumps({"choices": [{"message": {"content": "Draft a plan."}}]} if code == 200 else {"error": "x"}).encode()
            self.send_response(code)
            if code == 429:
                self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def test_retries_rate_limits_and_server_errors():
    srv, calls = _server([429, 503])
    try:
        c = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", backoff=0.01)
        assert c.chat([{"role": "user", "content": "hi"}]) == "Draft a plan."
        assert calls == [429, 503, 200]
    finally:
        srv.shutdown()


def test_gives_up_after_retries_and_does_not_retry_auth_errors():
    srv, calls = _server([503] * 10)
    try:
        c = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", retries=2, backoff=0.01)
        with pytest.raises(LLMError, match="HTTP 503"):
            c.chat([{"role": "user", "content": "hi"}])
        assert len(calls) == 3
    finally:
        srv.shutdown()

    srv, calls = _server([401])
    try:
        c = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", backoff=0.01)
        with pytest.raises(LLMError, match="API key"):
            c.chat([{"role": "user", "content": "hi"}])
        assert calls == [401]
    finally:
        srv.shutdown()
