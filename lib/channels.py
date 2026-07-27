"""Channels — a light logical grouping of projects (like YouTube channels).

Projects are never moved on disk (that would break every cached path); a channel
is just a label. The mapping lives in channels.json at the project root:

    { "channels": ["Superfruits", "History facts"],
      "assign":   { "video01": "Superfruits", "video02": "Superfruits" } }

`channels` is the ordered list of channel names (kept even when empty, so a fresh
channel doesn't vanish); `assign` maps a project id to its one channel. A project
with no entry is simply unassigned ("No channel"). Every read tolerates a missing
or malformed file by falling back to empty, so this can never break the dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "channels.json"
IMG_DIR = ROOT / "cache" / "channel_images"     # uploaded channel avatars live here

# The optional per-channel profile fields. Everything here is additive: an older
# channels.json without a "meta" block loads exactly as before, so upgrading can
# never drop a channel, an assignment, or a project.
META_FIELDS = ("image", "email", "description")


def _clean_meta(raw) -> dict:
    """A {channel: {image,email,description}} map, sanitised to strings."""
    out = {}
    for name, m in (raw or {}).items():
        if not isinstance(m, dict):
            continue
        fields = {k: str(m.get(k) or "").strip() for k in META_FIELDS}
        if any(fields.values()):
            out[str(name)] = fields
    return out


def _load() -> dict:
    if FILE.exists():
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
            channels, seen = [], set()
            for c in d.get("channels", []):
                c = str(c).strip()
                if c and c not in seen:
                    seen.add(c)
                    channels.append(c)
            assign = {str(k): str(v).strip()
                      for k, v in (d.get("assign") or {}).items()
                      if str(v).strip()}
            # A name used in assign but missing from the list is still a channel.
            for name in assign.values():
                if name not in seen:
                    seen.add(name)
                    channels.append(name)
            return {"channels": channels, "assign": assign,
                    "meta": _clean_meta(d.get("meta"))}
        except Exception:
            pass
    return {"channels": [], "assign": {}, "meta": {}}


def _save(d: dict) -> None:
    FILE.write_text(json.dumps({"channels": d["channels"], "assign": d["assign"],
                                "meta": d.get("meta", {})},
                               indent=2) + "\n", encoding="utf-8")


def data(valid_pids=None) -> dict:
    """The whole map. If `valid_pids` is given, assignments for projects that no
    longer exist are pruned (and persisted) so the file can't accumulate stale
    entries after deletes."""
    d = _load()
    if valid_pids is not None:
        pids = set(valid_pids)
        pruned = {k: v for k, v in d["assign"].items() if k in pids}
        if pruned != d["assign"]:
            d["assign"] = pruned
            _save(d)
    return d


def names() -> list[str]:
    return _load()["channels"]


def of(pid: str) -> str:
    """The channel a project belongs to, or '' for none."""
    return _load()["assign"].get(pid, "")


def create(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("channel name is empty")
    d = _load()
    if name not in d["channels"]:
        d["channels"].append(name)
        _save(d)
    return d


def assign(pid: str, channel: str) -> dict:
    """Move a project to a channel (creating it if new); '' clears the channel."""
    pid = (pid or "").strip()
    channel = (channel or "").strip()
    if not pid:
        raise ValueError("no project id")
    d = _load()
    if channel:
        if channel not in d["channels"]:
            d["channels"].append(channel)
        d["assign"][pid] = channel
    else:
        d["assign"].pop(pid, None)
    _save(d)
    return d


def rename(old: str, new: str) -> dict:
    old, new = (old or "").strip(), (new or "").strip()
    if not new:
        raise ValueError("new channel name is empty")
    d = _load()
    out, seen = [], set()
    for c in d["channels"]:
        c = new if c == old else c
        if c not in seen:
            seen.add(c)
            out.append(c)
    d["channels"] = out
    d["assign"] = {k: (new if v == old else v) for k, v in d["assign"].items()}
    if old in d.get("meta", {}) and old != new:      # carry the profile across
        d["meta"][new] = d["meta"].pop(old)
    _save(d)
    return d


def remove(name: str) -> dict:
    """Delete the channel label. Its projects are just unassigned — never deleted."""
    name = (name or "").strip()
    d = _load()
    d["channels"] = [c for c in d["channels"] if c != name]
    d["assign"] = {k: v for k, v in d["assign"].items() if v != name}
    m = d.get("meta", {}).pop(name, None)            # forget its profile too
    if m and m.get("image"):
        try:
            (IMG_DIR / m["image"]).unlink()          # tidy the orphaned avatar
        except OSError:
            pass
    _save(d)
    return d


def forget_project(pid: str) -> None:
    """Drop a project's assignment (called when the project itself is deleted)."""
    d = _load()
    if pid in d["assign"]:
        d["assign"].pop(pid, None)
        _save(d)


# ─────────────────────────────────────────────────────── channel profile

def meta_of(name: str) -> dict:
    """The profile ({image,email,description}) for one channel, blanks if unset."""
    m = _load().get("meta", {}).get((name or "").strip(), {})
    return {k: m.get(k, "") for k in META_FIELDS}


def set_meta(name: str, *, email=None, description=None) -> dict:
    """Update a channel's email/description (only the fields passed). Ensures the
    channel exists so editing a freshly made channel always sticks. The image is
    set separately via set_image()."""
    name = (name or "").strip()
    if not name:
        raise ValueError("channel name is empty")
    d = _load()
    if name not in d["channels"]:
        d["channels"].append(name)
    cur = d["meta"].get(name, {k: "" for k in META_FIELDS})
    cur = {k: cur.get(k, "") for k in META_FIELDS}
    if email is not None:
        cur["email"] = str(email).strip()
    if description is not None:
        cur["description"] = str(description).strip()
    d["meta"][name] = cur
    _save(d)
    return d


def set_image(name: str, raw: bytes, ext: str) -> str:
    """Save a channel avatar and record it; returns the stored filename. The old
    image (if any) is removed so avatars don't pile up, and a random token in the
    name busts any browser cache after a re-upload."""
    import hashlib
    import uuid
    name = (name or "").strip()
    if not name:
        raise ValueError("channel name is empty")
    ext = (ext or "png").lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg", "webp", "gif"):
        ext = "png"
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    fname = f"ch_{key}_{uuid.uuid4().hex[:8]}.{ext}"
    (IMG_DIR / fname).write_bytes(raw)

    d = _load()
    if name not in d["channels"]:
        d["channels"].append(name)
    cur = {k: d["meta"].get(name, {}).get(k, "") for k in META_FIELDS}
    old = cur.get("image")
    cur["image"] = fname
    d["meta"][name] = cur
    _save(d)
    if old and old != fname:
        try:
            (IMG_DIR / old).unlink()
        except OSError:
            pass
    return fname
