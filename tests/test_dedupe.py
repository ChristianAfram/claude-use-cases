import os
import random
from pathlib import Path

import numpy as np
import pytest

from usecase_gen.__main__ import DEFAULT_THRESHOLDS
from usecase_gen.dedupe import DedupeIndex, HashEmbedder, HttpEmbedder, load_existing, normalize, parse_list, verify
from usecase_gen.export import next_entry_number, next_volume, render_volume, volume_number
from usecase_gen.generate import Taxonomy, clean_output, plan_categories
from usecase_gen.llm import OpenAICompatClient

VOL1 = """# Claude Use Cases

## Contents

1. [Writing](#writing)
2. [Coding](#coding)

## Writing

1. Draft a polite follow-up email to a client about an overdue invoice.
2. Summarize a 40-page board deck into five bullet points.

## Coding

3. Debug a flaky integration test that only fails on CI.
"""

VOL2 = """# Claude Use Cases — Volume 2

## Table of Contents
1. Ops

## Ops
4. Plan a warehouse layout that cuts picking time by 20%.
  5.   Translate a safety manual into plain Spanish.
not an entry
10 Missing dot should not parse.
"""


@pytest.fixture
def lists_dir(tmp_path: Path) -> Path:
    (tmp_path / "claude-use-cases.md").write_text(VOL1, encoding="utf-8")
    (tmp_path / "claude-use-cases-vol2.md").write_text(VOL2, encoding="utf-8")
    (tmp_path / "unrelated.md").write_text("99999. Not a list file.\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------- parsing


def test_parse_skips_contents_and_non_entries(lists_dir: Path):
    entries = parse_list(lists_dir / "claude-use-cases.md")
    assert [e.number for e in entries] == [1, 2, 3]
    assert entries[0].text == "Draft a polite follow-up email to a client about an overdue invoice."

    entries2 = parse_list(lists_dir / "claude-use-cases-vol2.md")
    assert [(e.number, e.text) for e in entries2] == [
        (4, "Plan a warehouse layout that cuts picking time by 20%."),
        (5, "Translate a safety manual into plain Spanish."),
    ]


def test_load_existing_only_reads_list_files(lists_dir: Path):
    entries = load_existing(lists_dir)
    assert len(entries) == 5
    assert all(e.source.startswith("claude-use-cases") for e in entries)


def test_parse_handles_bom_and_crlf(tmp_path: Path):
    p = tmp_path / "claude-use-cases-vol9.md"
    p.write_bytes("﻿7. Outline a podcast episode.\r\n8. Review a lease.\r\n".encode("utf-8"))
    assert [(e.number, e.text) for e in parse_list(p)] == [(7, "Outline a podcast episode."), (8, "Review a lease.")]


# ---------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    "a,b",
    [
        ("Draft a client email.", "draft a client email"),
        ("Draft  a **client** email!", "Draft a client email."),
        ("Summarize the team’s “Q3” notes.", "Summarize the team's \"Q3\" notes"),
        ("Plan a roll-out — fast.", "plan a roll-out fast"),
    ],
)
def test_normalize_equivalent(a, b):
    assert normalize(a) == normalize(b)


def test_normalize_keeps_meaningful_differences():
    assert normalize("Draft a client email.") != normalize("Draft a vendor email.")
    assert "roll-out" in normalize("Plan a roll-out.")


# ---------------------------------------------------------------- similarity threshold


class FixedEmbedder:
    """Returns pre-set vectors so threshold boundaries can be tested exactly."""

    name = "fixed"

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table

    def embed(self, texts):
        return np.stack([self.table[t] for t in texts]).astype(np.float32)


def _vec_at_cos(c: float) -> np.ndarray:
    return np.array([c, np.sqrt(1 - c * c)], dtype=np.float32)


def test_threshold_is_strictly_greater_than():
    base = np.array([1.0, 0.0], dtype=np.float32)
    emb = FixedEmbedder({"existing one.": base})
    idx = DedupeIndex(emb, threshold=0.88)
    idx.add_existing(["existing one."])
    assert not idx.check("candidate a.", _vec_at_cos(0.885)).ok
    assert idx.check("candidate b.", _vec_at_cos(0.875)).ok
    v = idx.check("candidate c.", _vec_at_cos(0.95))
    assert (v.ok, v.reason, v.match) == (False, "near_dup_existing", "existing one.")


KNOWN_NEAR_DUP = (
    "Draft a polite follow-up email to a client about an overdue invoice.",
    "Draft a polite follow-up email to a client regarding an overdue invoice.",
)
KNOWN_DISTINCT = (
    "Draft a polite follow-up email to a client about an overdue invoice.",
    "Draft a grant proposal budget narrative for a rural health clinic.",
)


def _real_embedder():
    url = os.environ.get("USECASE_GEN_EMBED_URL")
    if not url:
        pytest.skip("set USECASE_GEN_EMBED_URL (e.g. http://localhost:8081/v1) to test a real embedding model")
    return HttpEmbedder(OpenAICompatClient(url))


@pytest.fixture(params=["hash", "http"])
def embedder_threshold(request):
    """Each embedder is checked at the CLI's default threshold for it."""
    if request.param == "hash":
        return HashEmbedder(), DEFAULT_THRESHOLDS["hash"]
    return _real_embedder(), DEFAULT_THRESHOLDS["http"]


def test_known_near_duplicate_pair_is_rejected(embedder_threshold):
    emb, thr = embedder_threshold
    idx = DedupeIndex(emb, threshold=thr)
    idx.add_existing([KNOWN_NEAR_DUP[0]])
    cand = KNOWN_NEAR_DUP[1]
    v = idx.check(cand, emb.embed([cand])[0])
    assert not v.ok and v.reason == "near_dup_existing" and v.score > thr


def test_known_distinct_pair_is_accepted(embedder_threshold):
    emb, thr = embedder_threshold
    idx = DedupeIndex(emb, threshold=thr)
    idx.add_existing([KNOWN_DISTINCT[0]])
    cand = KNOWN_DISTINCT[1]
    v = idx.check(cand, emb.embed([cand])[0])
    assert v.ok and v.score < thr


def test_exact_duplicates_rejected_existing_and_new():
    emb = HashEmbedder()
    idx = DedupeIndex(emb, threshold=0.88)
    idx.add_existing(["Review a lease."])
    v = idx.check("review a LEASE", emb.embed(["review a LEASE"])[0])
    assert (v.ok, v.reason) == (False, "exact_dup_existing")

    text = "Forecast monthly churn for a subscription bakery."
    vec = emb.embed([text])[0]
    assert idx.check(text, vec).ok
    idx.accept(text, vec)
    v = idx.check(text.upper(), emb.embed([text.upper()])[0])
    assert (v.ok, v.reason) == (False, "exact_dup_new")


def test_near_duplicate_within_new_batch_rejected(embedder_threshold):
    emb, thr = embedder_threshold
    idx = DedupeIndex(emb, threshold=thr)
    a, b = KNOWN_NEAR_DUP
    idx.accept(a, emb.embed([a])[0])
    v = idx.check(b, emb.embed([b])[0])
    assert (v.ok, v.reason) == (False, "near_dup_new")


def test_verify_flags_violations_and_passes_clean_sets(embedder_threshold):
    emb, thr = embedder_threshold
    bad = verify([KNOWN_NEAR_DUP[1]], [KNOWN_NEAR_DUP[0]], emb, thr)
    assert not bad["clean"] and bad["new_entries_over_threshold_vs_existing"] == 1
    good = verify([KNOWN_DISTINCT[1]], [KNOWN_DISTINCT[0]], emb, thr)
    assert good["clean"]


# ---------------------------------------------------------------- numbering + volumes


def test_numbering_offset_continues_from_highest(lists_dir: Path):
    assert next_entry_number(load_existing(lists_dir)) == 6


def test_numbering_offset_uses_max_not_count(tmp_path: Path):
    (tmp_path / "claude-use-cases-vol3.md").write_text("2000. A.\n2001. B.\n", encoding="utf-8")
    (tmp_path / "claude-use-cases.md").write_text("1. C.\n", encoding="utf-8")
    assert next_entry_number(load_existing(tmp_path)) == 2002
    assert next_volume(tmp_path) == 4


def test_numbering_empty_dir_starts_at_one(tmp_path: Path):
    assert next_entry_number(load_existing(tmp_path)) == 1
    assert next_volume(tmp_path) == 1


@pytest.mark.parametrize(
    "name,num",
    [("claude-use-cases.md", 1), ("claude-use-cases-vol2.md", 2), ("claude-use-cases-3.md", 3),
     ("claude-use-cases_vol12.md", 12), ("claude-use-cases-volume5.md", 5), ("claude-use-cases-notes.md", None)],
)
def test_volume_number(name, num):
    assert volume_number(Path(name)) == num


def test_next_volume(lists_dir: Path):
    assert next_volume(lists_dir) == 3


def test_rendered_volume_roundtrips_through_parser(tmp_path: Path):
    cats = {"Healthcare": [f"Draft note {i} for a clinic." for i in range(3)],
            "Gaming": [f"Design level {i} for a puzzle game." for i in range(2)]}
    md = render_volume(3, 101, cats)
    p = tmp_path / "claude-use-cases-vol3.md"
    p.write_text(md, encoding="utf-8")
    entries = parse_list(p)
    assert [e.number for e in entries] == [101, 102, 103, 104, 105]
    assert "## Contents" in md and "- [Healthcare](#healthcare) — 101–103" in md
    assert "- [Gaming](#gaming) — 104–105" in md


# ---------------------------------------------------------------- generation helpers


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Draft a churn-recovery email for a solo founder with no budget.", "Draft a churn-recovery email for a solo founder with no budget."),
        ('1. "draft a churn-recovery email for a solo founder"', "Draft a churn-recovery email for a solo founder."),
        ("Draft a one-page brief, e.g. for a board.", "Draft a one-page brief, e.g. for a board."),
    ],
)
def test_clean_output_accepts(raw, expected):
    assert clean_output(raw, "Draft") == (expected, "")


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("", "empty"),
        ("Draft an email.", "too_short"),
        ("Draft " + "very " * 25 + "long email.", "too_long"),
        ("Write a churn-recovery email for a solo founder.", "wrong_verb"),
        ("Draft: Write a churn-recovery email for a solo founder.", "wrong_verb"),
        ("Drafting a churn-recovery email for a solo founder.", "wrong_verb"),
        ("Draft a churn email for a founder. Then send it to the team.", "multi_sentence"),
    ],
)
def test_clean_output_rejects(raw, reason):
    assert clean_output(raw, "Draft") == (None, reason)


def test_plan_categories_quotas_and_least_used_first():
    tax = Taxonomy(verbs=["A", "B"], domains=["D1", "D2", "D3"], contexts=[f"c{i}" for i in range(60)])
    used = {("A", "D1", "c0")}
    q = plan_categories(tax, 150, 100, used, random.Random(0))
    assert sum(q.values()) == 150 and sorted(q.values()) == [50, 100]
    assert "D1" not in q  # most-used domain is skipped
