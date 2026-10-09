"""Parse existing lists, normalize entries, and reject exact / near duplicates.

Existing claude-use-cases*.md files are opened read-only and never written.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .llm import OpenAICompatClient

LIST_GLOB = "claude-use-cases*.md"
ENTRY_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*?)\s*#*\s*$")
CONTENTS_TITLES = {"contents", "table of contents", "toc", "index"}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


@dataclass(frozen=True)
class Entry:
    number: int
    text: str
    source: str  # file name


def list_files(lists_dir: Path) -> list[Path]:
    return sorted(p for p in Path(lists_dir).glob(LIST_GLOB) if p.is_file())


def parse_list(path: Path) -> list[Entry]:
    """Return every numbered entry ("123. Sentence.") outside a Contents section."""
    entries: list[Entry] = []
    in_contents = False
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            h = HEADING_RE.match(line)
            if h:
                in_contents = h.group(1).strip().strip(":").lower() in CONTENTS_TITLES
                continue
            if in_contents:
                continue
            m = ENTRY_RE.match(line)
            if m:
                entries.append(Entry(int(m.group(1)), m.group(2), path.name))
    return entries


def load_existing(lists_dir: Path) -> list[Entry]:
    out: list[Entry] = []
    for p in list_files(lists_dir):
        out.extend(parse_list(p))
    return out


def normalize(text: str) -> str:
    """Canonical form for exact-duplicate checks: case, punctuation, quotes and spacing don't count."""
    t = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    t = re.sub(r"\*\*|__|`", "", t)  # markdown emphasis
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = re.sub(r"(?<!\w)['-]|['-](?!\w)", " ", t)  # quotes/dashes not inside a word
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- embedders


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, offline lexical embedder (word + char-trigram feature hashing).

    Useful for tests and for runs without an embedding server. It only catches
    lexical near-duplicates; use a real embedding model for paraphrases.
    """

    name = "hash-v2"

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        norm = normalize(text)
        words = norm.split()
        feats = [f"w:{w}" for w in words]
        padded = f" {norm} "
        feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _unit(out)


class HttpEmbedder:
    """Embeddings from an OpenAI-compatible /v1/embeddings endpoint (llama-server --embeddings)."""

    def __init__(self, client: OpenAICompatClient, batch_size: int = 64, prefix: str = ""):
        self.client = client
        self.batch_size = batch_size
        self.prefix = prefix  # task prefix some models expect, e.g. "clustering: " for nomic-embed
        self.name = f"http:{client.base_url}:{client.model}:{prefix}"

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self.client.embed([self.prefix + t for t in texts[i:i + self.batch_size]]))
        return _unit(np.asarray(rows, dtype=np.float32))


class CachedEmbedder:
    """Disk cache so the existing corpus is embedded once, not every run."""

    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        slug = hashlib.sha1(inner.name.encode()).hexdigest()[:12]
        self.path_vecs = Path(cache_dir) / f"emb-{slug}.npy"
        self.path_keys = Path(cache_dir) / f"emb-{slug}.json"
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        if self.path_vecs.exists() and self.path_keys.exists():
            keys = json.loads(self.path_keys.read_text(encoding="utf-8"))
            vecs = np.load(self.path_vecs)
            if len(keys) == len(vecs):
                self._cache = dict(zip(keys, vecs))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if missing:
            first_text = {k: t for k, t in zip(reversed(keys), reversed(texts))}
            vecs = self.inner.embed([first_text[k] for k in missing])
            self._cache.update(zip(missing, vecs))
            self._dirty = True
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

    def save(self) -> None:
        if not self._dirty:
            return
        self.path_vecs.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        np.save(self.path_vecs, np.stack([self._cache[k] for k in keys]))
        self.path_keys.write_text(json.dumps(keys), encoding="utf-8")
        self._dirty = False


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


# ---------------------------------------------------------------- index


@dataclass
class Verdict:
    ok: bool
    reason: str = ""  # "", "exact_dup_existing", "exact_dup_new", "near_dup_existing", "near_dup_new"
    score: float = 0.0
    match: str = ""


class DedupeIndex:
    """Exact (normalized) + cosine-similarity index over existing and accepted entries.

    A candidate is rejected if its normalized text is already present, or if its
    cosine similarity to ANY indexed entry is strictly greater than `threshold`.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.88):
        self.embedder = embedder
        self.threshold = threshold
        self._norm: dict[str, bool] = {}  # normalized text -> is_new
        self._texts: list[str] = []
        self._is_new: list[bool] = []
        self._mat: np.ndarray | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def _append(self, texts: list[str], vecs: np.ndarray, is_new: bool) -> None:
        if len(texts) == 0:
            return
        if self._mat is None:
            self._mat = np.zeros((max(1024, 2 * len(texts)), vecs.shape[1]), dtype=np.float32)
        need = self._n + len(texts)
        if need > len(self._mat):
            grown = np.zeros((max(need, 2 * len(self._mat)), self._mat.shape[1]), dtype=np.float32)
            grown[: self._n] = self._mat[: self._n]
            self._mat = grown
        self._mat[self._n:need] = vecs
        self._n = need
        for t in texts:
            self._texts.append(t)
            self._is_new.append(is_new)
            self._norm.setdefault(normalize(t), is_new)

    def add_existing(self, texts: list[str]) -> None:
        self._append(texts, self.embedder.embed(texts) if texts else np.zeros((0, 0)), is_new=False)

    def check(self, text: str, vec: np.ndarray) -> Verdict:
        n = normalize(text)
        if n in self._norm:
            return Verdict(False, "exact_dup_new" if self._norm[n] else "exact_dup_existing", 1.0, text)
        if self._n == 0:
            return Verdict(True)
        sims = self._mat[: self._n] @ vec
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score > self.threshold:
            return Verdict(False, "near_dup_new" if self._is_new[i] else "near_dup_existing", score, self._texts[i])
        return Verdict(True, score=score, match=self._texts[i])

    def accept(self, text: str, vec: np.ndarray) -> None:
        self._append([text], vec.reshape(1, -1), is_new=True)


def verify(new_texts: list[str], existing_texts: list[str], embedder: Embedder, threshold: float) -> dict:
    """Independent full re-check of a finished batch. Returns violation counts and worst pairs."""
    norm_existing = {normalize(t) for t in existing_texts}
    norm_new = [normalize(t) for t in new_texts]
    exact_existing = sum(n in norm_existing for n in norm_new)
    exact_new = len(norm_new) - len(set(norm_new))

    new_vecs = embedder.embed(new_texts) if new_texts else np.zeros((0, 1), np.float32)
    max_vs_existing, worst_existing = 0.0, None
    best = np.full(len(new_texts), -1.0, dtype=np.float32)
    for start in range(0, len(existing_texts) if new_texts else 0, 8192):  # chunked: bounded memory
        chunk = existing_texts[start:start + 8192]
        sims = new_vecs @ embedder.embed(chunk).T
        best = np.maximum(best, sims.max(axis=1))
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        if worst_existing is None or sims[i, j] > max_vs_existing:
            max_vs_existing = float(sims[i, j])
            worst_existing = (new_texts[i], chunk[j])
    over_existing = int((best > threshold).sum())

    max_within, worst_within, over_within = 0.0, None, 0
    if len(new_texts) > 1:
        sims = new_vecs @ new_vecs.T
        np.fill_diagonal(sims, -1.0)
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        max_within = float(sims[i, j])
        worst_within = (new_texts[i], new_texts[j])
        over_within = int((np.triu(sims, 1) > threshold).sum())

    return {
        "exact_dup_vs_existing": exact_existing,
        "exact_dup_within_new": exact_new,
        "new_entries_over_threshold_vs_existing": over_existing,
        "pairs_over_threshold_within_new": over_within,
        "max_sim_vs_existing": round(max_vs_existing, 4),
        "max_sim_within_new": round(max_within, 4),
        "worst_pair_vs_existing": worst_existing,
        "worst_pair_within_new": worst_within,
        "clean": exact_existing == 0 and exact_new == 0 and over_existing == 0 and over_within == 0,
    }
