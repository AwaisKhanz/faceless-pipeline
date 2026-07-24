"""The persistent job store: lifecycle, FIFO order, atomic persistence, restart
recovery. No HTTP, no pipeline — pure state."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import jobs as J   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '} {label:<54}{'' if ok else repr(got)}")
        if not ok:
            bad += 1

    tmp = Path(tempfile.mkdtemp())
    path = tmp / "queue.json"

    print("Persistent job store\n")

    st = J.JobStore(path, save_every=0)          # save every time, for the test
    a = st.add("proj-a", "generate")
    b = st.add("proj-b", "source", args={"redo": [1, 2]})
    check("add returns a queued job", a.status, J.QUEUED)
    check("ids are unique", a.id != b.id, True)
    check("FIFO seq increments", (a.seq, b.seq), (1, 2))
    check("next_queued is the oldest", st.next_queued().id, a.id)
    check("args are stored", st.get(b.id).args, {"redo": [1, 2]})

    # lifecycle: run a, it stamps started; finish it, stamps ended
    st.update(a.id, status=J.RUNNING)
    check("running stamps started", st.get(a.id).started is not None, True)
    check("next_queued skips the running one", st.next_queued().id, b.id)
    st.append_log(a.id, "hello")
    st.append_log(a.id, "world")
    check("log accumulates", [x["text"] for x in st.get(a.id).log], ["hello", "world"])
    st.update(a.id, status=J.DONE)
    check("done stamps ended", st.get(a.id).ended is not None, True)
    check("done is terminal, not active", a.id not in [j.id for j in st.active()], True)

    # log cap
    for i in range(J.LOG_CAP + 50):
        st.append_log(b.id, f"line {i}")
    check("log is capped at LOG_CAP", len(st.get(b.id).log), J.LOG_CAP)

    # ── persistence round-trips ─────────────────────────────────────────────
    print("\n  persistence + restart recovery:")
    st.update(b.id, status=J.RUNNING, label="sourcing")   # b is 'running' on disk
    check("the file was written", path.exists(), True)

    st2 = J.JobStore(path)
    st2.load()
    check("all jobs reload", {j.id for j in st2.jobs()}, {a.id, b.id})
    check("the done job stays done", st2.get(a.id).status, J.DONE)
    check("a job left RUNNING becomes INTERRUPTED (no more 'start from 0')",
          st2.get(b.id).status, J.INTERRUPTED)
    check("reloaded label survived", st2.get(b.id).label, "sourcing")
    check("seq continues after reload", st2.add("c", "voice").seq, 3)

    # atomic write: a valid JSON file, no leftover .tmp
    check("no leftover temp file", (tmp / "queue.json.tmp").exists(), False)

    # remove + clear_finished
    print("\n  removal:")
    st2.update(a.id, status=J.DONE)
    removed = st2.remove(b.id)
    check("remove deletes a job", removed and st2.get(b.id) is None, True)
    n = st2.clear_finished()
    check("clear_finished drops terminal jobs", n >= 1, True)
    check("a queued/active job is kept by clear_finished",
          any(j.status == J.QUEUED for j in st2.jobs()), True)

    # a corrupt file must not crash load
    print("\n  robustness:")
    (tmp / "bad.json").write_text("{ not json", encoding="utf-8")
    st3 = J.JobStore(tmp / "bad.json")
    st3.load()
    check("a corrupt store loads as empty, no crash", st3.jobs(), [])

    # ── scheduler: FIFO, concurrency, cancel, error, resume ─────────────────
    print("\n  scheduler runs the queue:")

    def wait_until(fn, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if fn():
                return True
            time.sleep(0.02)
        return False

    # FIFO order + a done outcome
    s1 = J.JobStore(tmp / "s1.json", save_every=0)
    order = []

    def rec(job):
        order.append(job.project)
        time.sleep(0.05)
    sch = J.Scheduler(s1, {"x": rec}, max_concurrent=1, poll=0.05)
    for p in ("a", "b", "c"):
        sch.enqueue(p, "x")
    ok = wait_until(lambda: all(j.status == J.DONE for j in s1.jobs()) and len(s1.jobs()) == 3)
    check("all three ran and finished", ok, True)
    check("they ran in FIFO order", order, ["a", "b", "c"])
    sch.stop()

    # concurrency: max 2 in flight at once
    s2 = J.JobStore(tmp / "s2.json", save_every=0)
    live = {"now": 0, "peak": 0}
    live_lock = __import__("threading").Lock()

    def busy(job):
        with live_lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.1)
        with live_lock:
            live["now"] -= 1
    sch2 = J.Scheduler(s2, {"x": busy}, max_concurrent=2, poll=0.05)
    for _ in range(4):
        sch2.enqueue("p", "x")
    wait_until(lambda: all(j.status == J.DONE for j in s2.jobs()) and len(s2.jobs()) == 4)
    check("max_concurrent is honoured (peak == 2)", live["peak"], 2)
    sch2.stop()

    # cancel a queued job (never runs) and error propagation
    s3 = J.JobStore(tmp / "s3.json", save_every=0)
    ran = []

    def slow(job):
        for _ in range(50):
            if job.cancel:
                return
            time.sleep(0.02)
        ran.append(job.id)

    def boom(job):
        raise RuntimeError("kaboom")
    sch3 = J.Scheduler(s3, {"slow": slow, "boom": boom}, max_concurrent=1, poll=0.05)
    j_run = sch3.enqueue("p", "slow")
    j_queued = sch3.enqueue("p", "slow")
    j_err = sch3.enqueue("p", "boom")
    wait_until(lambda: s3.get(j_run.id).status == J.RUNNING)
    sch3.cancel(j_queued.id)                     # cancel the one still waiting
    check("a queued job cancelled before it runs", s3.get(j_queued.id).status, J.CANCELED)
    sch3.cancel(j_run.id)                        # ask the running one to stop
    wait_until(lambda: s3.get(j_run.id).status == J.CANCELED)
    check("a running job stops when cancelled", s3.get(j_run.id).status, J.CANCELED)
    wait_until(lambda: s3.get(j_err.id).status == J.ERROR)
    check("a crashing worker becomes ERROR, queue keeps going",
          s3.get(j_err.id).status, J.ERROR)
    check("the crash message is kept", "kaboom" in s3.get(j_err.id).error, True)
    sch3.stop()

    # resume: an INTERRUPTED job is re-queued and run on startup
    s4 = J.JobStore(tmp / "s4.json", save_every=0)
    dead = s4.add("p", "x")
    s4.update(dead.id, status=J.INTERRUPTED)
    seen = []
    sch4 = J.Scheduler(s4, {"x": lambda job: seen.append(job.id)},
                       max_concurrent=1, resume=True, poll=0.05)
    wait_until(lambda: s4.get(dead.id).status == J.DONE)
    check("an interrupted job auto-resumes on start", seen, [dead.id])
    sch4.stop()

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
