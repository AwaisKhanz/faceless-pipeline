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
    stage: str = ""                              # the studio's step name
    lang: str | None = None
    done: int = 0
    total: int = 0
    eta: float | None = None
    rate: float | None = None
    error: str = ""
    outputs: list = field(default_factory=list)
    steps: list = field(default_factory=list)    # [{name, lang, seconds, items}]
    log: list = field(default_factory=list)      # [{"t": epoch, "text": str}]
    cancel: bool = False                         # set to ask a running job to stop
    created: float = 0.0
    started: float | None = None
    step_started: float | None = None
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


class Scheduler:
    """Runs queued jobs in FIFO order, up to `max_concurrent` at a time.

    Decoupled from HTTP and the pipeline: it is handed a `run_map` of
    kind -> callable(job) that does the actual work (blocking). The studio wraps
    each real worker so that, before it runs, a thread-local 'current job' is set,
    letting the existing set_job / log / progress helpers report into THIS job.

    A worker signals its outcome by setting the job's status (done / error /
    approve). If it returns without doing so, the job is marked done. On start,
    any INTERRUPTED job (the server died mid-run) is re-queued — auto-resume —
    and because every pipeline step is cached, it picks up where it left off.
    """

    def __init__(self, store: JobStore, run_map: dict, max_concurrent: int = 1,
                 resume: bool = True, poll: float = 0.5, gate=None):
        self.store = store
        self.run_map = run_map
        self.max_concurrent = max(1, int(max_concurrent))
        # gate(job, running_jobs) -> bool: may this job start given what's already
        # running? Lets the studio keep two GPU-heavy jobs from overlapping while
        # still letting a network-bound job run alongside one. None = no gate.
        self.gate = gate
        self._poll = poll
        self._threads: "dict[str, threading.Thread]" = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = False
        if resume:
            for j in store.jobs():
                if j.status == INTERRUPTED:
                    store.update(j.id, status=QUEUED, cancel=False)
        self._dispatcher = threading.Thread(target=self._loop, daemon=True)
        self._dispatcher.start()

    # ── public ──────────────────────────────────────────────────────────────
    def enqueue(self, project: str, kind: str, args: dict | None = None,
                auto: bool = False) -> Job:
        job = self.store.add(project, kind, args, auto=auto)
        self._wake.set()
        return job

    def cancel(self, jid: str) -> bool:
        """Stop a job. A queued job is dropped; a running one is asked to stop
        (its worker sees the cancel flag between items and bows out)."""
        job = self.store.get(jid)
        if job is None or job.status in TERMINAL:
            return False
        if job.status == QUEUED:
            self.store.update(jid, status=CANCELED, force_save=True)
        else:
            self.store.update(jid, cancel=True, force_save=True)
        self._wake.set()
        return True

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._threads.values() if t.is_alive())

    def stop(self) -> None:                      # for tests / shutdown
        self._stopped = True
        self._wake.set()

    # ── internals ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stopped:
            self._wake.wait(timeout=self._poll)
            self._wake.clear()
            self._dispatch()

    def _dispatch(self) -> None:
        with self._lock:
            self._threads = {jid: t for jid, t in self._threads.items()
                             if t.is_alive()}
            running = [j for j in (self.store.get(jid) for jid in self._threads)
                       if j is not None]
            while len(self._threads) < self.max_concurrent:
                # FIFO, but backfill: if the oldest queued job is held back by the
                # gate (e.g. a GPU job while another GPU job runs), a later job on
                # a free resource may start ahead of it, so the queue never stalls.
                nxt = None
                for job in self.store.jobs():
                    if job.status != QUEUED:
                        continue
                    if self.gate and not self.gate(job, running):
                        continue
                    nxt = job
                    break
                if nxt is None:
                    break
                self.store.update(nxt.id, status=RUNNING, cancel=False,
                                  started=time.time(), force_save=True)
                t = threading.Thread(target=self._run, args=(nxt.id,), daemon=True)
                self._threads[nxt.id] = t
                running.append(self.store.get(nxt.id))
                t.start()

    def _run(self, jid: str) -> None:
        job = self.store.get(jid)
        fn = self.run_map.get(job.kind) if job else None
        try:
            if fn is None:
                self.store.update(jid, status=ERROR,
                                  error=f"no worker for kind '{job.kind}'",
                                  force_save=True)
            else:
                fn(job)
        except BaseException as e:               # a crash must not wedge the queue
            self.store.update(jid, status=ERROR,
                              error=str(e) or type(e).__name__, force_save=True)
        finally:
            cur = self.store.get(jid)
            if cur is not None and cur.status == RUNNING:
                # worker finished without declaring an outcome → treat as done,
                # unless it was asked to cancel.
                self.store.update(
                    jid, status=(CANCELED if cur.cancel else DONE), force_save=True)
            self._wake.set()                     # a slot freed — try the next
