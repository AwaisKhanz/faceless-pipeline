"""Persistent job queue for the studio.

A single source of truth for every unit of background work — generate, source,
voice, render — modelled as a Job with a lifecycle and progress, and stored on
disk so the queue and its progress survive a browser refresh AND a server
restart. Today a killed server forgot everything and "started from 0"; a Job
that was running when the server died is reloaded as `interrupted`, ready for the
scheduler to resume.

This module is pure state + persistence. It knows nothing about HTTP or the video
pipeline, which is what makes it small and unit-testable on its own. The studio
server (Phase 2) drives it; the workers report into a Job instead of one global
dict.

Concurrency: every public method takes a re-entrant lock, so the HTTP threads and
the worker threads can touch the store safely. Writes are debounced and atomic
(temp file + os.replace) so frequent log/progress updates never hammer the disk
or leave a half-written file behind.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

# ── lifecycle ───────────────────────────────────────────────────────────────
QUEUED = "queued"          # waiting its turn
RUNNING = "running"        # a worker is on it
APPROVE = "approve"        # finished its step, waiting for the user (e.g. review)
DONE = "done"              # finished for good
ERROR = "error"            # failed
CANCELED = "canceled"      # the user stopped it
INTERRUPTED = "interrupted"  # the server died mid-run; may be resumed

ACTIVE = frozenset({RUNNING, APPROVE})           # occupies the worker / the UI
TERMINAL = frozenset({DONE, ERROR, CANCELED})    # will never run again
LOG_CAP = 600              # keep only the last N log lines per job (memory + disk)


@dataclass
class Job:
    """One unit of background work, JSON-round-trippable."""
    id: str
    project: str
    kind: str                                    # generate|source|voice|render|…
    seq: int                                     # FIFO order across the whole store
    status: str = QUEUED
    args: dict = field(default_factory=dict)     # the kwargs the worker needs
    auto: bool = False                           # part of an auto-pipeline chain
    label: str = ""
    stage: str = ""
    done: int = 0
    total: int = 0
    eta: float | None = None
    rate: float | None = None
    error: str = ""
    outputs: list = field(default_factory=list)
    log: list = field(default_factory=list)      # [{"t": epoch, "text": str}]
    created: float = 0.0
    started: float | None = None
    ended: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        # Ignore unknown keys so an older/newer file never crashes the load.
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


class JobStore:
    """Thread-safe, disk-backed collection of Jobs."""

    def __init__(self, path, save_every: float = 1.5):
        self.path = str(path)
        self._save_every = save_every
        self._lock = threading.RLock()
        self._jobs: "dict[str, Job]" = {}
        self._seq = 0
        self._dirty = False
        self._last_save = 0.0

    # ── creation / lookup ───────────────────────────────────────────────────
    def add(self, project: str, kind: str, args: dict | None = None,
            auto: bool = False, status: str = QUEUED) -> Job:
        with self._lock:
            self._seq += 1
            job = Job(id=uuid.uuid4().hex[:8], project=project, kind=kind,
                      seq=self._seq, status=status, args=dict(args or {}),
                      auto=auto, created=time.time())
            self._jobs[job.id] = job
            self.save(force=True)
            return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def jobs(self) -> list[Job]:
        """Every job, oldest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.seq)

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self.jobs() if j.status in ACTIVE]

    def next_queued(self) -> Job | None:
        """The oldest job still waiting to run."""
        with self._lock:
            for j in self.jobs():
                if j.status == QUEUED:
                    return j
            return None

    # ── mutation ────────────────────────────────────────────────────────────
    def update(self, jid: str, force_save: bool = False, **fields) -> Job | None:
        """Set fields on a job. Status changes stamp started/ended and always
        persist immediately; plain progress/label updates are debounced."""
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return None
            status_changed = "status" in fields and fields["status"] != job.status
            for k, v in fields.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            if status_changed:
                if job.status == RUNNING and not job.started:
                    job.started = time.time()
                if job.status in TERMINAL or job.status == INTERRUPTED:
                    job.ended = time.time()
            self._dirty = True
            self.save(force=force_save or status_changed)
            return job

    def append_log(self, jid: str, text: str) -> None:
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return
            job.log.append({"t": time.time(), "text": str(text)})
            del job.log[:-LOG_CAP]
            self._dirty = True
            self.save()

    def remove(self, jid: str) -> bool:
        with self._lock:
            existed = self._jobs.pop(jid, None) is not None
            if existed:
                self.save(force=True)
            return existed

    def clear_finished(self) -> int:
        """Drop DONE/ERROR/CANCELED jobs; returns how many were removed."""
        with self._lock:
            gone = [jid for jid, j in self._jobs.items() if j.status in TERMINAL]
            for jid in gone:
                del self._jobs[jid]
            if gone:
                self.save(force=True)
            return len(gone)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, force: bool = False) -> None:
        """Atomically write the store to disk. Debounced unless `force`, so a
        burst of log lines does not turn into a burst of disk writes."""
        with self._lock:
            if not self._dirty and not force:
                return
            now = time.time()
            if not force and (now - self._last_save) < self._save_every:
                return
            data = {"seq": self._seq,
                    "jobs": [j.to_dict() for j in self.jobs()]}
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = f"{self.path}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.path)       # atomic on POSIX and Windows
                self._dirty = False
                self._last_save = now
            except Exception:
                pass                             # never let a save crash a job

    def load(self) -> None:
        """Read the store from disk. A job left RUNNING (the server died mid-work)
        becomes INTERRUPTED so the scheduler can decide to resume it — it never
        silently vanishes, which is what made a restart 'start from 0'."""
        with self._lock:
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return
            self._jobs = {}
            for d in data.get("jobs", []):
                try:
                    job = Job.from_dict(d)
                except Exception:
                    continue
                if job.status in ACTIVE:         # was running/awaiting when we died
                    job.status = INTERRUPTED
                    job.ended = job.ended or time.time()
                self._jobs[job.id] = job
            self._seq = max([data.get("seq", 0),
                             *[j.seq for j in self._jobs.values()]] or [0])
            self._dirty = False
