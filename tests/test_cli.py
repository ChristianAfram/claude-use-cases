"""End-to-end: run the real CLI against a stub OpenAI-compatible server (chat + embeddings)."""

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from usecase_gen.__main__ import main
from usecase_gen.dedupe import normalize, parse_list

EXISTING_NEAR_DUP_TARGET = "Draft the weekly sales pipeline summary for the regional team lead."
EXISTING = f"""# Claude Use Cases

## Contents

1. [Misc](#misc)

## Misc

1. {EXISTING_NEAR_DUP_TARGET}
2. Translate a safety manual into plain Spanish.
"""
EXISTING_VOL2 = "# Volume 2\n\n## Misc\n\n41. Plan a warehouse layout that cuts picking time by 20%.\n"


def _stub_vector(text: str) -> list[float]:
    # Ignores the first word, so "<Verb> the weekly sales pipeline ..." collides with the
    # existing entry: different normalized text (not an exact dup) but cosine 1.0 (near dup).
    rest = " ".join(normalize(text).split()[1:])
    seed = int.from_bytes(hashlib.sha1(rest.encode()).digest()[:4], "little")
    return np.random.default_rng(seed).standard_normal(256).tolist()


class StubState:
    def __init__(self):
        self.calls = 0
        self.first_by_verb: dict[str, str] = {}
        self.lock = threading.Lock()


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                data = [{"index": i, "embedding": _stub_vector(t)} for i, t in enumerate(req["input"])]
                return self._send({"data": data})
            prompt = req["messages"][-1]["content"]
            verb = re.search(r"Task verb: (.+)", prompt).group(1)
            domain = re.search(r"Domain: (.+)", prompt).group(1)
            context = re.search(r"Context: (.+)", prompt).group(1)
            with state.lock:
                state.calls += 1
                n = state.calls
                first = state.first_by_verb.get(verb)
            if n % 11 == 0:
                out = f"{verb} " + "a very long rambling sentence " * 5
            elif n % 13 == 0:
                out = f"{verb} the weekly sales pipeline summary for the regional team lead."
            elif n % 7 == 0 and first:
                out = first.upper()  # exact duplicate after normalization
            else:
                out = f"<think>hmm</think>{verb} a {domain.lower()} checklist {context} (#{n})."
                with state.lock:
                    state.first_by_verb.setdefault(verb, out.split("</think>")[1])
            return self._send({"choices": [{"message": {"role": "assistant", "content": out}}]})

    return Handler


@pytest.fixture
def stub_server():
    state = StubState()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", state
    srv.shutdown()


def test_cli_writes_clean_volume_and_continues_numbering(tmp_path: Path, stub_server):
    url, state = stub_server
    (tmp_path / "claude-use-cases.md").write_text(EXISTING, encoding="utf-8")
    (tmp_path / "claude-use-cases-vol2.md").write_text(EXISTING_VOL2, encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.md")}

    rc = main(["--count", "250", "--lists-dir", str(tmp_path), "--llm-url", url, "--seed", "1", "--workers", "4"])
    assert rc == 0

    # Existing lists untouched.
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data

    out = tmp_path / "claude-use-cases-vol3.md"
    md = out.read_text(encoding="utf-8")
    entries = parse_list(out)
    assert [e.number for e in entries] == list(range(42, 42 + 250))
    assert len({normalize(e.text) for e in entries}) == 250
    assert md.count("\n## ") == 4  # Contents + 3 categories (100, 100, 50)
    assert re.search(r"- \[.+\]\(#.+\) — 42–141", md)

    run_dir = next((tmp_path / ".usecase_gen" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["verification"]["clean"]
    rej = summary["rejections"]
    assert rej["too_long"] > 0 and rej["near_dup_existing"] > 0 and rej["exact_dup_new"] > 0
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "rejected near_dup_existing" in log and "verification:" in log

    ledger = (tmp_path / ".usecase_gen" / "used_triples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 250

    # Second run: next volume, numbering continues, previously used triples are not reused.
    rc = main(["--count", "100", "--lists-dir", str(tmp_path), "--llm-url", url, "--seed", "2"])
    assert rc == 0
    entries4 = parse_list(tmp_path / "claude-use-cases-vol4.md")
    assert entries4[0].number == 292 and entries4[-1].number == 391
    rows = [json.loads(x) for x in (tmp_path / ".usecase_gen" / "used_triples.jsonl").read_text(encoding="utf-8").splitlines()]
    triples = [(r["verb"], r["domain"], r["context"]) for r in rows]
    assert len(triples) == len(set(triples)) == 350


def test_cli_fails_cleanly_when_llm_unreachable(tmp_path: Path):
    (tmp_path / "claude-use-cases.md").write_text(EXISTING, encoding="utf-8")
    rc = main(["--count", "10", "--lists-dir", str(tmp_path), "--llm-url", "http://127.0.0.1:9/v1", "--embedder", "hash"])
    assert rc == 4
    run_dir = next((tmp_path / ".usecase_gen" / "runs").iterdir())
    assert "Every LLM call in a wave failed" in (run_dir / "run.log").read_text(encoding="utf-8")
    assert not (tmp_path / "claude-use-cases-vol2.md").exists()
