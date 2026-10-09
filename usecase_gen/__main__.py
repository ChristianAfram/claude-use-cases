"""CLI: uv run python -m usecase_gen --count 1000"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .dedupe import CachedEmbedder, DedupeIndex, HashEmbedder, HttpEmbedder, list_files, load_existing, verify
from .export import next_entry_number, next_volume, render_volume, volume_path, write_volume
from .generate import append_ledger, generate, load_ledger, load_taxonomy, plan_categories
from .llm import LLMError, OpenAICompatClient

log = logging.getLogger("usecase_gen")
DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomy.yaml")
# Similarity scales are model-specific: the lexical hash fallback scores rewordings lower than a semantic model.
DEFAULT_THRESHOLDS = {"http": 0.88, "hash": 0.80}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="usecase_gen", description=__doc__)
    p.add_argument("--count", type=int, default=1000, help="entries to generate (default 1000)")
    p.add_argument("--per-category", type=int, default=100, help="entries per category/domain (default 100)")
    p.add_argument("--lists-dir", type=Path, default=Path.cwd(), help="folder with claude-use-cases*.md (read-only); the new volume is written here")
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--state-dir", type=Path, default=None, help="ledger, embedding cache, run logs (default: <lists-dir>/.usecase_gen)")
    p.add_argument("--llm-url", default="http://localhost:8080/v1")
    p.add_argument("--llm-model", default="local")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--embedder", choices=["http", "hash"], default="http",
                   help="http = OpenAI-compatible /embeddings (default); hash = offline lexical fallback")
    p.add_argument("--embed-url", default=None, help="embeddings endpoint (default: --llm-url)")
    p.add_argument("--embed-model", default="local")
    p.add_argument("--embed-prefix", default="", help='text prepended before embedding, e.g. "clustering: " for nomic-embed')
    p.add_argument("--threshold", type=float, default=None,
                   help=f"reject if cosine similarity > this (default {DEFAULT_THRESHOLDS['http']}; hash: {DEFAULT_THRESHOLDS['hash']})")
    p.add_argument("--workers", type=int, default=4, help="parallel LLM requests; match llama-server -np")
    p.add_argument("--max-attempts", type=int, default=None, help="LLM call budget (default 5 x count)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sim", nargs=2, metavar=("A", "B"), help="print cosine similarity of two sentences and exit (threshold calibration)")
    args = p.parse_args(argv)
    if args.threshold is None:
        args.threshold = DEFAULT_THRESHOLDS[args.embedder]
    return args


def build_embedder(args: argparse.Namespace, cache_dir: Path) -> CachedEmbedder:
    if args.embedder == "hash":
        inner = HashEmbedder()
    else:
        inner = HttpEmbedder(OpenAICompatClient(args.embed_url or args.llm_url, args.embed_model, args.api_key),
                             prefix=args.embed_prefix)
    return CachedEmbedder(inner, cache_dir)


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args = parse_args(argv)
    try:
        return run(args)
    except (LLMError, RuntimeError, ValueError) as e:
        if log.handlers:
            log.error("%s", e)
        else:
            print(f"error: {e}", file=sys.stderr)
        return 4


def run(args: argparse.Namespace) -> int:
    lists_dir = args.lists_dir.resolve()
    state_dir = (args.state_dir or lists_dir / ".usecase_gen").resolve()
    embedder = build_embedder(args, state_dir / "cache")

    if args.sim:
        a, b = embedder.embed(list(args.sim))
        score = float(np.dot(a, b))
        print(f"cosine={score:.4f}  threshold={args.threshold}  -> {'REJECT (near-duplicate)' if score > args.threshold else 'accept'}")
        embedder.save()
        return 0

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = state_dir / "runs" / run_id
    setup_logging(run_dir)
    t0 = time.time()
    rng = random.Random(args.seed)

    tax = load_taxonomy(args.taxonomy)
    log.info("taxonomy: %d verbs x %d domains x %d contexts = %d triples",
             len(tax.verbs), len(tax.domains), len(tax.contexts), len(tax.verbs) * len(tax.domains) * len(tax.contexts))

    files = list_files(lists_dir)
    existing = load_existing(lists_dir)
    per_file = Counter(e.source for e in existing)
    for f in files:
        log.info("existing list %s: %d entries", f.name, per_file[f.name])
    log.info("existing total: %d entries in %d files", len(existing), len(files))

    vol = next_volume(lists_dir)
    start = next_entry_number(existing)
    out_path = volume_path(lists_dir, vol)
    if out_path.exists():
        log.error("%s already exists; refusing to overwrite", out_path.name)
        return 2
    log.info("target: %s, %d entries numbered %d-%d", out_path.name, args.count, start, start + args.count - 1)

    ledger_path = state_dir / "used_triples.jsonl"
    used = load_ledger(ledger_path)
    quotas = plan_categories(tax, args.count, args.per_category, used, rng)
    log.info("categories: %s", ", ".join(f"{d} ({n})" for d, n in quotas.items()))

    log.info("embedding existing corpus with %s ...", embedder.name)
    existing_texts = [e.text for e in existing]
    index = DedupeIndex(embedder, args.threshold)
    index.add_existing(existing_texts)
    embedder.save()
    log.info("index ready: %d vectors, threshold %.2f", len(index), args.threshold)

    client = OpenAICompatClient(args.llm_url, args.llm_model, args.api_key)
    rejected_f = open(run_dir / "rejected.jsonl", "w", encoding="utf-8", newline="\n")
    accepted_f = open(run_dir / "accepted.jsonl", "w", encoding="utf-8", newline="\n")
    try:
        res = generate(
            tax, quotas,
            chat=lambda msgs: client.chat(msgs, args.temperature, args.max_tokens),
            index=index, embedder=embedder, used=used, rng=rng,
            workers=args.workers, max_attempts=args.max_attempts,
            on_reject=lambda r: (rejected_f.write(json.dumps(r, ensure_ascii=False) + "\n"), rejected_f.flush()),
            on_accept=lambda d, a: (accepted_f.write(json.dumps({"domain": d, "text": a.text, "triple": a.triple}, ensure_ascii=False) + "\n"), accepted_f.flush()),
        )
    finally:
        rejected_f.close()
        accepted_f.close()
        embedder.save()

    log.info("attempts: %d, accepted: %d, rejected: %d", res.attempts, res.total, sum(res.rejections.values()))
    for reason, n in res.rejections.most_common():
        log.info("  rejected %-20s %d", reason, n)

    summary = {"run_id": run_id, "volume": vol, "attempts": res.attempts, "accepted": res.total,
               "rejections": dict(res.rejections), "threshold": args.threshold, "embedder": embedder.name}
    if res.total < args.count:
        log.error("only %d/%d accepted (exhausted: %s). No volume written; accepted entries kept in %s",
                  res.total, args.count, res.exhausted or "attempt budget", run_dir / "accepted.jsonl")
        _write_summary(run_dir, summary)
        return 1

    new_texts = [a.text for d in quotas for a in res.accepted[d]]
    report = verify(new_texts, existing_texts, embedder, args.threshold)
    summary["verification"] = report
    log.info("verification: exact dups vs existing=%d, within new=%d; over threshold vs existing=%d, within new=%d; "
             "max sim vs existing=%.4f, within new=%.4f",
             report["exact_dup_vs_existing"], report["exact_dup_within_new"],
             report["new_entries_over_threshold_vs_existing"], report["pairs_over_threshold_within_new"],
             report["max_sim_vs_existing"], report["max_sim_within_new"])
    if not report["clean"]:
        log.error("verification FAILED; no volume written. See %s", run_dir / "summary.json")
        _write_summary(run_dir, summary)
        return 3

    content = render_volume(vol, start, {d: [a.text for a in res.accepted[d]] for d in quotas})
    path = write_volume(lists_dir, vol, content)
    n = start
    rows = []
    for d in quotas:
        for a in res.accepted[d]:
            rows.append({"verb": a.triple[0], "domain": d, "context": a.triple[2], "volume": vol, "number": n})
            n += 1
    append_ledger(ledger_path, rows)
    summary["output"] = str(path)
    _write_summary(run_dir, summary)
    log.info("wrote %s (%d entries, %d-%d) in %.0fs; run log: %s", path.name, res.total, start, n - 1, time.time() - t0, run_dir)
    return 0


def _write_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
