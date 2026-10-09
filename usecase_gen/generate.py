"""Sample unused (verb, domain, context) triples and have the LLM write one sentence per triple."""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from .dedupe import DedupeIndex, Embedder

log = logging.getLogger("usecase_gen")

Triple = tuple[str, str, str]  # (verb, domain, context)
MAX_WORDS = 20
MIN_WORDS = 6


@dataclass(frozen=True)
class Taxonomy:
    verbs: list[str]
    domains: list[str]
    contexts: list[str]


def load_taxonomy(path: Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tax = Taxonomy(*(list(dict.fromkeys(str(x).strip() for x in data.get(k) or [])) for k in ("verbs", "domains", "contexts")))
    for name, items in (("verbs", tax.verbs), ("domains", tax.domains), ("contexts", tax.contexts)):
        if not items:
            raise ValueError(f"{path}: '{name}' is empty")
    bad = [v for v in tax.verbs if not re.fullmatch(r"[A-Za-z]+", v)]
    if bad:
        raise ValueError(f"{path}: verbs must be single words, got {bad}")
    return tax


# ---------------------------------------------------------------- ledger of used triples


def load_ledger(path: Path) -> set[Triple]:
    used: set[Triple] = set()
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                used.add((r["verb"], r["domain"], r["context"]))
    return used


def append_ledger(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- planning


def plan_categories(tax: Taxonomy, count: int, per_category: int, used: set[Triple], rng: random.Random) -> dict[str, int]:
    """Pick the least-used domains first; each gets `per_category` slots (last one may get fewer)."""
    n_cats = -(-count // per_category)
    capacity = len(tax.verbs) * len(tax.contexts)
    if per_category > capacity:
        raise ValueError(f"per_category={per_category} exceeds verb x context combos per domain ({capacity})")
    if n_cats > len(tax.domains):
        raise ValueError(f"--count {count} needs {n_cats} categories but taxonomy has {len(tax.domains)} domains")
    usage = Counter(d for _, d, _ in used)
    ranked = sorted(tax.domains, key=lambda d: (usage[d], rng.random()))[:n_cats]
    chosen = [d for d in tax.domains if d in ranked]  # keep taxonomy order in the output
    quotas = {d: per_category for d in chosen}
    quotas[chosen[-1]] = count - per_category * (n_cats - 1)
    return quotas


# ---------------------------------------------------------------- prompt + output cleaning


def build_messages(verb: str, domain: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You write concise, concrete use cases for an AI assistant. Reply with exactly one sentence and nothing else.",
        },
        {
            "role": "user",
            "content": (
                "Write one realistic use case.\n\n"
                f"Task verb: {verb}\nDomain: {domain}\nContext: {context}\n\n"
                "Rules:\n"
                f"- One plain sentence, at most {MAX_WORDS} words, ending with a period.\n"
                f'- Start with the word "{verb}" (imperative).\n'
                "- Name a concrete artifact or outcome: a specific document, dataset, plan, or message.\n"
                "- Reflect both the domain and the context.\n"
                "- Do not mention AI, Claude, or the assistant. No quotes, lists, labels, or preamble."
            ),
        },
    ]


_ABBREV_RE = re.compile(r"\b(e\.g|i\.e|etc|vs|approx|incl)\.", re.IGNORECASE)


def clean_output(raw: str, verb: str) -> tuple[str | None, str]:
    """Return (sentence, "") or (None, reject_reason)."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return None, "empty"
    s = lines[0]
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", s)
    s = re.sub(r"^(?:use case|sentence|answer|output)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(" \"'*`“”‘’")
    s = re.sub(r"\s+", " ", s).rstrip(" .!?;:,") + "."
    if len(s) < 2:
        return None, "empty"
    s = s[0].upper() + s[1:]
    words = s.split()
    if len(words) > MAX_WORDS:
        return None, "too_long"
    if len(words) < MIN_WORDS:
        return None, "too_short"
    # The verb must open the sentence as a word, not as a label ("Critique: Analyze ...").
    if not re.match(rf"{re.escape(verb)}(?![\w:])", s, re.IGNORECASE):
        return None, "wrong_verb"
    if re.search(r"[.!?]\s+\S", _ABBREV_RE.sub("", s[:-1])):
        return None, "multi_sentence"
    return s, ""


# ---------------------------------------------------------------- main loop


@dataclass
class Accepted:
    text: str
    triple: Triple
    nearest_score: float


@dataclass
class GenResult:
    accepted: dict[str, list[Accepted]]
    rejections: Counter = field(default_factory=Counter)
    attempts: int = 0
    exhausted: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.accepted.values())


ChatFn = Callable[[list[dict]], str]


def generate(
    tax: Taxonomy,
    quotas: dict[str, int],
    chat: ChatFn,
    index: DedupeIndex,
    embedder: Embedder,
    used: set[Triple],
    rng: random.Random,
    workers: int = 4,
    max_attempts: int | None = None,
    on_reject: Callable[[dict], None] | None = None,
    on_accept: Callable[[str, Accepted], None] | None = None,
) -> GenResult:
    pools: dict[str, list[tuple[str, str]]] = {}
    for d in quotas:
        pool = [(v, c) for v in tax.verbs for c in tax.contexts if (v, d, c) not in used]
        rng.shuffle(pool)
        pools[d] = pool
    res = GenResult(accepted={d: [] for d in quotas})
    max_attempts = max_attempts or 5 * sum(quotas.values())
    wave_cap = max(workers * 4, 8)

    def need(d: str) -> int:
        return quotas[d] - len(res.accepted[d])

    def call(triple: Triple) -> tuple[Triple, str | None, str]:
        try:
            return triple, chat(build_messages(*triple)), ""
        except Exception as e:  # noqa: BLE001 - any client failure is logged and counted
            return triple, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        while res.attempts < max_attempts:
            open_cats = [d for d in quotas if need(d) > 0 and pools[d]]
            if not open_cats:
                break
            # Round-robin across categories; never request more than a category still needs.
            wave: list[Triple] = []
            taken = Counter()
            while len(wave) < wave_cap and len(wave) + res.attempts < max_attempts:
                progressed = False
                for d in open_cats:
                    if taken[d] < need(d) and pools[d] and len(wave) < wave_cap:
                        v, c = pools[d].pop()
                        wave.append((v, d, c))
                        taken[d] += 1
                        progressed = True
                if not progressed:
                    break
            if not wave:
                break
            results = list(pool_exec.map(call, wave))
            res.attempts += len(wave)

            errors = [r for r in results if r[1] is None]
            if len(errors) == len(results):
                raise RuntimeError(f"Every LLM call in a wave failed; last error: {errors[-1][2]}")

            cleaned: list[tuple[Triple, str]] = []
            for triple, raw, err in results:
                if raw is None:
                    res.rejections["llm_error"] += 1
                    pools[triple[1]].insert(0, (triple[0], triple[2]))  # not the triple's fault: retry later
                    _emit(on_reject, triple, raw, "llm_error", detail=err)
                    continue
                text, reason = clean_output(raw, triple[0])
                if text is None:
                    res.rejections[reason] += 1
                    _emit(on_reject, triple, raw, reason)
                else:
                    cleaned.append((triple, text))

            if not cleaned:
                continue
            vecs = embedder.embed([t for _, t in cleaned])
            for (triple, text), vec in zip(cleaned, vecs):
                verdict = index.check(text, vec)
                if verdict.ok:
                    index.accept(text, vec)
                    acc = Accepted(text, triple, round(verdict.score, 4))
                    res.accepted[triple[1]].append(acc)
                    if on_accept:
                        on_accept(triple[1], acc)
                else:
                    res.rejections[verdict.reason] += 1
                    _emit(on_reject, triple, text, verdict.reason, score=verdict.score, match=verdict.match)

            done = res.total
            log.info(
                "progress %d/%d accepted, %d attempts, %d rejected",
                done, sum(quotas.values()), res.attempts, sum(res.rejections.values()),
            )

    res.exhausted = [d for d in quotas if need(d) > 0 and not pools[d]]
    return res


def _emit(cb: Callable[[dict], None] | None, triple: Triple, text: str | None, reason: str, **extra) -> None:
    if cb:
        row = {"reason": reason, "verb": triple[0], "domain": triple[1], "context": triple[2], "text": text}
        row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in extra.items()})
        cb(row)
