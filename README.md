# claude-use-cases

Lists of Claude use cases (`claude-use-cases*.md`) plus `usecase_gen`, a local CLI that writes the next
volume: new use cases from a **verb × domain × context** taxonomy, deduped (exact + embedding similarity)
against every existing list.

## Quick start (Windows 11, PowerShell)

```powershell
# 1. Chat model on :8080 (4 parallel slots)
llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080 -np 4 -c 8192

# 2. Embedding model on :8081 (a real embedding model, not the chat model)
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --port 8081 --embeddings --pooling mean -c 2048 -ub 2048

# 3. Generate 1,000 entries into the folder holding your lists
uv run python -m usecase_gen --count 1000 --embed-url http://localhost:8081/v1 --embed-prefix "clustering: "
```

Output: `claude-use-cases-volN.md` (N = highest existing volume + 1), numbered from the highest existing
entry number + 1, with a Contents section and 100 entries per category (categories = taxonomy domains).
Existing `claude-use-cases*.md` files are only read; the CLI refuses to overwrite any file.

## How it works

1. Parse every `claude-use-cases*.md` in `--lists-dir` (lines like `123. Sentence.`; Contents sections skipped).
2. Pick `count / 100` domains, least-used first, and shuffle the unused `(verb, context)` pairs for each.
3. Ask the LLM for one sentence per triple: plain, ≤ 20 words, starting with the verb.
4. Reject on format, exact match (normalized: case, punctuation, quotes, spacing ignored), or cosine
   similarity **> 0.88** to any existing *or already accepted* entry.
5. Repeat until every category is full, re-verify the whole batch independently, then write the volume.
   Nothing is written if the target isn't reached or verification finds a single violation.

## Flags worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--count` | 1000 | entries to generate |
| `--per-category` | 100 | entries per domain section |
| `--lists-dir` | current dir | where lists are read and the new volume is written |
| `--llm-url` / `--embed-url` | `http://localhost:8080/v1` / same | any OpenAI-compatible server |
| `--threshold` | 0.88 (`hash`: 0.80) | reject if cosine similarity is strictly greater |
| `--embedder` | `http` | `hash` = offline lexical fallback, catches rewording but not paraphrase |
| `--workers` | 4 | parallel LLM calls; match `llama-server -np` |
| `--sim "A" "B"` | | print the similarity of two sentences, for calibrating the threshold |

## Threshold calibration

0.88 is only meaningful for a specific embedding model. Measured with nomic-embed-text-v1.5 (Q8_0), against
*"Draft a polite follow-up email to a client about an overdue invoice."*:

| Candidate | no prefix | `clustering: ` |
|---|---|---|
| …to a client **regarding** an overdue invoice. | 0.999 | 0.999 |
| …to a **customer** about an overdue invoice. | 0.965 | 0.971 |
| **Write a courteous reminder to a customer whose invoice is past due.** (paraphrase) | 0.810 ✗ missed | 0.897 ✓ rejected |
| …to a **vendor** about a **delayed shipment**. (different task) | 0.839 | 0.871 kept |
| Draft a grant proposal budget narrative for a rural health clinic. | 0.506 | 0.660 kept |

With nomic, use `--embed-prefix "clustering: "`. With another model, check a few pairs from your own lists
with `--sim` first, then tune `--threshold`. `rejected.jsonl` records the score and nearest match for every
near-duplicate, which tells you whether the line sits in the right place.

## State and logs (`<lists-dir>/.usecase_gen/`)

- `used_triples.jsonl`: ledger of triples already used in a written volume, so later runs don't repeat them. Keep it.
- `cache/`: embedding cache, so the existing corpus is embedded once per model. Safe to delete.
- `runs/<timestamp>/`: `run.log` (rejection counts by reason plus the verification report), `rejected.jsonl`
  (every rejected sentence with its reason, nearest match and score), `accepted.jsonl`, `summary.json`.

Rejection reasons: `llm_error`, `empty`, `too_short`, `too_long`, `wrong_verb`, `multi_sentence`,
`exact_dup_existing`, `exact_dup_new`, `near_dup_existing`, `near_dup_new`.

## Tests

```powershell
uv run pytest
```

The tests run offline: an end-to-end CLI run against a stub OpenAI-compatible server, plus unit tests for
parsing, normalization, the threshold and numbering. Set `USECASE_GEN_EMBED_URL` to also run the known
near-duplicate pair against your real embedding model.
