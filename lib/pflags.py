"""Per-project flags — small pieces of state that live outside the project's own
files. Right now that's just "uploaded to YouTube", a manual mark the user sets
once a video has been published, so the dashboard can highlight it.

Stored centrally in project_flags.json at the project root:

    { "video01": { "uploaded": true, "uploaded_at": "2026-07-28T10:00:00Z" } }

Every read tolerates a missing or malformed file by falling back to empty, so a
flag can never break the dashboard. Purely additive: a project with no entry is
simply "not uploaded".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "project_flags.json"


def _load() -> dict:
    if FILE.exists():
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
            out = {}
            for pid, f in (d or {}).items():
                if isinstance(f, dict):
                    out[str(pid)] = {"uploaded": bool(f.get("uploaded")),
                                     "uploaded_at": str(f.get("uploaded_at") or "")}
            return out
        except Exception:
            pass
    return {}


def _save(d: dict) -> None:
    FILE.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")


def data(valid_pids=None) -> dict:
    """The whole map. If `valid_pids` is given, entries for projects that no
    longer exist are pruned (and persisted) so the file can't accumulate stale
    rows after deletes."""
    d = _load()
    if valid_pids is not None:
        pids = set(valid_pids)
        pruned = {k: v for k, v in d.items() if k in pids}
        if pruned != d:
            _save(pruned)
            d = pruned
    return d


def get(pid: str) -> dict:
    """This project's flags, with sane blanks when unset."""
    f = _load().get((pid or "").strip(), {})
    return {"uploaded": bool(f.get("uploaded")),
            "uploaded_at": f.get("uploaded_at", "")}


def set_uploaded(pid: str, value: bool) -> dict:
    """Mark (or unmark) a project as uploaded to YouTube, stamping the time."""
    pid = (pid or "").strip()
    if not pid:
        raise ValueError("no project id")
    d = _load()
    if value:
        d[pid] = {"uploaded": True,
                  "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    else:
        d.pop(pid, None)                       # unmarked == absent
    _save(d)
    return get(pid)


def forget(pid: str) -> None:
    """Drop a project's flags (called when the project itself is deleted)."""
    d = _load()
    if (pid or "").strip() in d:
        d.pop(pid.strip(), None)
        _save(d)
