"""Flagging heuristics for review."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


def flag_lines(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Return lines and reasons for review."""

    flags: List[Dict[str, str]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        reasons = []
        if "[unclear]" in line:
            reasons.append("contains [unclear]")
        if re.search(r"\S{25,}", line):
            reasons.append("long token")
        non_alpha = sum(1 for ch in line if not ch.isalpha() and not ch.isspace())
        if line:
            ratio = non_alpha / len(line)
            if ratio > 0.35:
                reasons.append("high non-alphabetic ratio")
        if "?" in line or re.search(r"[!@#$%^&*_=+]{2,}", line):
            reasons.append("suspicious punctuation")
        if reasons:
            flags.append({"line_no": idx, "reason": "; ".join(reasons)})

    return {"lines": flags}


def write_flags(flags: Dict[str, List[Dict[str, str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flags, indent=2), encoding="utf-8")
