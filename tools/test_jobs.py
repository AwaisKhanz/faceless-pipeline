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

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
