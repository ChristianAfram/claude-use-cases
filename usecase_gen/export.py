"""Write the next volume: claude-use-cases-volN.md, numbered after the highest existing entry."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .dedupe import Entry, list_files

VOL_RE = re.compile(r"^claude-use-cases(?:[-_ ]?vol(?:ume)?)?[-_ ]?(\d+)?\.md$", re.IGNORECASE)


def volume_number(path: Path) -> int | None:
    """claude-use-cases.md -> 1, claude-use-cases-vol7.md -> 7, claude-use-cases-3.md -> 3."""
    m = VOL_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def next_volume(lists_dir: Path) -> int:
    nums = [n for p in list_files(lists_dir) if (n := volume_number(p)) is not None]
    return max(nums, default=0) + 1


def next_entry_number(existing: list[Entry]) -> int:
    return max((e.number for e in existing), default=0) + 1


def volume_path(lists_dir: Path, vol: int) -> Path:
    return Path(lists_dir) / f"claude-use-cases-vol{vol}.md"


def _anchor(title: str) -> str:
    # GitHub-style heading anchor.
    a = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"\s", "-", a)


def render_volume(vol: int, start: int, categories: dict[str, list[str]], today: date | None = None) -> str:
    """categories: ordered {category: [sentence, ...]}. Numbering is continuous across categories."""
    today = today or date.today()
    total = sum(len(v) for v in categories.values())
    end = start + total - 1
    lines = [
        f"# Claude Use Cases — Volume {vol}",
        "",
        f"{total:,} use cases, numbered {start:,}–{end:,}. Generated {today.isoformat()} by usecase_gen.",
        "",
        "## Contents",
        "",
    ]
    n = start
    for cat, items in categories.items():
        lines.append(f"- [{cat}](#{_anchor(cat)}) — {n}–{n + len(items) - 1}")
        n += len(items)
    n = start
    for cat, items in categories.items():
        lines += ["", f"## {cat}", ""]
        for text in items:
            lines.append(f"{n}. {text}")
            n += 1
    return "\n".join(lines) + "\n"


def write_volume(lists_dir: Path, vol: int, content: str) -> Path:
    path = volume_path(lists_dir, vol)
    # "x" mode: refuse to overwrite anything, including an existing list.
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return path
