"""Veo video generation: availability, prompt, async submit/poll/decode, cap."""
import base64
import sys
import tempfile
import time as _time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import veo   # noqa: E402
from lib import llm   # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '} {label:<54}{'' if ok else repr(got)}")
        if not ok:
            bad += 1

    print("Veo video generation\n")
    # available() delegates to llm.vertex_ready; stub it so this suite tests the
    # delegation, not whether google-auth happens to be installed here.
    llm.vertex_ready = lambda cfg: bool((cfg or {}).get("vertex_project"))
    check("no project -> unavailable", veo.available({}), False)
    check("a project -> available", veo.available({"vertex_project": "p"}), True)
    p = veo.prompt_for("a rocket launch", {})
    check("prompt keeps the subject", "a rocket launch" in p)
    check("prompt adds a motion/footage style", "footage" in p or "motion" in p)

    # never actually sleep or wait real time
    llm._vertex_token = lambda sa: "TOK"
    veo.time = SimpleNamespace(sleep=lambda s: None, time=_time.time)

    b64 = base64.b64encode(b"MP4DATA").decode()
    calls = {"polls": 0}

    def fake_post(url, body, token):
        if url.endswith(":predictLongRunning"):
            calls["body"] = body
            return {"name": "projects/x/locations/y/operations/op1"}
        calls["polls"] += 1                       # fetchPredictOperation
        if calls["polls"] == 1:
            return {"done": False}                # still rendering
        return {"done": True, "response":
                {"videos": [{"bytesBase64Encoded": b64, "mimeType": "video/mp4"}]}}
    veo._post = fake_post

    cfg = {"vertex_project": "proj", "veo_location": "us-central1", "veo_seconds": 8}
    dest = Path(tempfile.mkdtemp()) / "v.mp4"
    waits = []
    out = veo.video("a rocket", cfg, dest, on_wait=lambda: waits.append(1))
    check("writes the mp4", out.exists() and out.stat().st_size > 0)
    check("asks for 16:9", calls["body"]["parameters"]["aspectRatio"], "16:9")
    check("asks for exactly ONE clip", calls["body"]["parameters"]["sampleCount"], 1)
    check("passes durationSeconds", calls["body"]["parameters"]["durationSeconds"], 8)
    check("no audio (the scene has its own narration)",
          calls["body"]["parameters"]["generateAudio"], False)
    check("polls until the operation is done", calls["polls"] >= 2, True)
    check("pings on_wait while it renders", len(waits) >= 1, True)

    # cache: a second call for the same dest must NOT hit the API
    def boom(*a, **k):
        raise AssertionError("should have used the cached clip")
    veo._post = boom
    check("a re-run reuses the cached clip", veo.video("a rocket", cfg, dest) == dest)

    # a safety-filtered result -> VeoError, not a broken file
    veo._post = lambda url, body, token: (
        {"name": "op"} if url.endswith("predictLongRunning")
        else {"done": True, "response": {"raiMediaFilteredReasons": ["blocked"]}})
    try:
        veo.video("x", cfg, Path(tempfile.mkdtemp()) / "v2.mp4")
        check("a filtered clip raises", False)
    except veo.VeoError:
        check("a filtered clip raises VeoError", True)

    # missing project -> VeoError before any network call
    try:
        veo.video("x", {"veo_location": "us-central1"},
                  Path(tempfile.mkdtemp()) / "v3.mp4")
        check("missing project raises", False)
    except veo.VeoError:
        check("missing project raises VeoError", True)

    # ── generate_videos: the capped manual path ─────────────────────────────
    print("\n  generate_videos is capped and marks clips as video:")
    import shutil
    from lib import pipeline as pl
    veo.available = lambda cfg: True

    def fake_video(prompt, cfg, dest, on_wait=None, should_cancel=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"MP4")
        return Path(dest)
    veo.video = fake_video

    tpid = "ZZZveotest"
    sdir = ROOT / "projects" / tpid / "sheets"
    sdir.mkdir(parents=True, exist_ok=True)
    sheet = sdir / f"{tpid}_main_script.md"
    sheet.write_text("<!-- main-lang: en -->\n# t\n", encoding="utf-8")
    try:
        scenes = [SimpleNamespace(n=1, query="a", narration="", media="VIDEO"),
                  SimpleNamespace(n=2, query="b", narration="", media="VIDEO"),
                  SimpleNamespace(n=3, query="c", narration="", media="VIDEO")]
        res = pl.generate_videos(scenes, sheet, {"vertex_project": "p", "veo_max": 2},
                                 [1, 2, 3], log=lambda *_: None)
        check("cap: generates only veo_max (2)", res["generated"], [1, 2])
        check("cap: holds back the rest", res["skipped"], [3])
        check("the asset is marked as video", res["assets"][1]["media"], "VIDEO")
        check("the asset source is veo", res["assets"][1]["src"], "veo")
        check("the asset path is an mp4", res["assets"][1]["path"].endswith(".mp4"), True)
    finally:
        shutil.rmtree(ROOT / "projects" / tpid, ignore_errors=True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
