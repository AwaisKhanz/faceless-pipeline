"""Imagen generation: availability, prompt building, request shape, caching."""
import base64
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import imagen as im   # noqa: E402
from lib import llm           # noqa: E402


def main() -> int:
    bad = 0

    def check(label, got, want=True):
        nonlocal bad
        ok = got == want
        print(f"  {'ok ' if ok else '!! '} {label:<52}{'' if ok else repr(got)}")
        if not ok:
            bad += 1

    print("Imagen generation\n")
    # available() delegates to llm.vertex_ready (a real config check, incl.
    # google-auth); stub it here so this suite tests the delegation, not the env.
    llm.vertex_ready = lambda cfg: bool((cfg or {}).get("vertex_project"))
    check("no project -> unavailable", im.available({}), False)
    check("a project -> available", im.available({"vertex_project": "p"}), True)

    p = im.prompt_for("data center servers", {})
    check("prompt keeps the subject", "data center servers" in p)
    check("prompt forces a photoreal look by default", "photorealistic" in p)
    check("prompt tells the model NOT to illustrate", "not an illustration" in p.lower())
    check("a scene that asks for illustration is left non-photo",
          "photorealistic" not in im.prompt_for("watercolour illustration of a fox", {}))
    check("a custom style overrides the default",
          "neon" in im.prompt_for("x", {"generate_style": "neon"}))

    # image(): mock the token and the HTTP predict call
    llm._vertex_token = lambda sa: "TOK"
    cap = {}
    tiny = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode()

    def fake_generate(url, body, token):
        cap.update(url=url, body=body, token=token)
        return {"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": tiny}}]}}]}
    im._generate = fake_generate

    cfg = {"vertex_project": "proj", "generate_location": "global",
           "vertex_service_account": ""}
    dest = Path(tempfile.mkdtemp()) / "g.png"
    out = im.image("a rocket launch", cfg, dest)
    check("writes the image file", out.exists() and out.stat().st_size > 0)
    check("endpoint targets the project", "projects/proj/locations/global" in cap["url"])
    check("endpoint uses generateContent (Gemini image API)",
          cap["url"].endswith(":generateContent"))
    check("endpoint targets the default Gemini image model",
          "gemini-2.5-flash-image" in cap["url"])
    check("asks for an IMAGE modality",
          "IMAGE" in cap["body"]["generationConfig"]["responseModalities"])
    check("asks for 16:9",
          cap["body"]["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")
    check("passes the bearer token", cap["token"], "TOK")

    # caching: a second call for the same dest must NOT hit the API
    def boom(*a, **k):
        raise AssertionError("should have used the cached file")
    im._generate = boom
    out2 = im.image("a rocket launch", cfg, dest)
    check("a re-run reuses the cached file, no second call", out2 == dest)

    # a text-only reply (no image part / safety) -> GenError, not a broken file
    im._generate = lambda url, body, token: {
        "candidates": [{"content": {"parts": [{"text": "sorry"}]},
                        "finishReason": "IMAGE_SAFETY"}]}
    try:
        im.image("x", cfg, Path(tempfile.mkdtemp()) / "g2.png")
        check("a reply with no image raises", False)
    except im.GenError:
        check("a reply with no image raises GenError", True)

    # missing project -> GenError before any network call
    try:
        im.image("x", {"generate_location": "global"},
                 Path(tempfile.mkdtemp()) / "g3.png")
        check("missing project raises", False)
    except im.GenError:
        check("missing project raises GenError", True)

    # ── rate limiting: back off on 429 instead of surrendering ──────────────
    print("\n  429 backoff + throttle:")
    import io
    import urllib.error
    from types import SimpleNamespace as _NS

    check("retryDelay is read from the server's 429 body",
          im._retry_delay('{"error":{"details":[{"retryDelay":"7s"}]}}', 1), 7.5)
    check("no retryDelay -> exponential backoff (2,4,8…)",
          im._retry_delay("{}", 3), 8.0)
    check("generate_workers sizes the concurrency gate",
          im._semaphore({"generate_workers": 2})._value, 2)

    def _http_error(code, body=b"{}"):
        return urllib.error.HTTPError("http://x", code, "busy", {}, io.BytesIO(body))

    # Don't actually sleep during the test; record what we'd have waited.
    saved_time, sleeps = im.time, []
    im.time = _NS(sleep=lambda s: sleeps.append(s), monotonic=saved_time.monotonic)
    rl_cfg = {"vertex_project": "proj", "generate_location": "global",
              "vertex_service_account": "", "generate_min_interval": 0,
              "generate_retries": 5, "generate_workers": 1}
    try:
        tries = {"n": 0}

        def flaky(url, body, token):
            tries["n"] += 1
            if tries["n"] <= 2:                      # busy the first two times
                raise _http_error(429, b'{"error":{"details":[{"retryDelay":"3s"}]}}')
            return {"candidates": [{"content": {"parts": [
                {"inlineData": {"data": tiny}}]}}]}
        im._generate = flaky
        notes = []
        out = im.image("retry me", rl_cfg,
                       Path(tempfile.mkdtemp()) / "r.png", log=notes.append)
        check("succeeds once the model frees up (3 attempts)",
              out.exists() and tries["n"] == 3)
        check("waited the server's retryDelay (3s + margin)", 3.5 in sleeps)
        check("surfaced a retry notice to the log", any("retry" in m for m in notes))

        im._generate = lambda u, b, t: (_ for _ in ()).throw(_http_error(429))
        try:
            im.image("always busy", {**rl_cfg, "generate_retries": 2},
                     Path(tempfile.mkdtemp()) / "r2.png")
            check("a persistent 429 eventually raises", False)
        except im.GenError as e:
            check("a persistent 429 raises GenError (mentions 429)", "429" in str(e))
    finally:
        im.time = saved_time

    # ── manual per-scene generation (review page path) ──────────────────────
    print("\n  manual generate_scenes (the review-page 'Generate' button):")
    from types import SimpleNamespace
    import shutil
    from lib import pipeline as pl

    im.available = lambda cfg: True
    made: list = []

    def fake_image(prompt, cfg, dest, log=None):
        if "boom" in prompt:
            raise im.GenError("safety filter")     # e.g. a real person
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"IMG")
        made.append(str(dest))
        return Path(dest)
    im.image = fake_image

    root = Path(__file__).resolve().parent.parent
    tpid = "ZZZgentest"
    sdir = root / "projects" / tpid / "sheets"
    sdir.mkdir(parents=True, exist_ok=True)
    sheet = sdir / f"{tpid}_main_script.md"
    sheet.write_text("<!-- main-lang: en -->\n# t\n", encoding="utf-8")
    try:
        scenes = [SimpleNamespace(n=1, query="a data center", narration="x", media="IMAGE"),
                  SimpleNamespace(n=2, query="boom person", narration="y", media="IMAGE")]
        res = pl.generate_scenes(scenes, sheet, {"vertex_project": "p"}, [1, 2],
                                 log=lambda *_: None)
        check("generated the good scene", res["generated"], [1])
        check("reported the refused scene, did not crash",
              [n for n, _ in res["failed"]], [2])
        check("the generated asset is tagged generated", res["assets"][1]["generated"], True)
        check("the generated asset's source is imagen", res["assets"][1]["src"], "imagen")
        prev = res["assets"][1]["path"]
        res2 = pl.generate_scenes(scenes, sheet, {"vertex_project": "p"}, [1],
                                  log=lambda *_: None)
        check("re-generating makes a NEW take (fresh file)",
              res2["assets"][1]["path"] != prev, True)
    finally:
        shutil.rmtree(root / "projects" / tpid, ignore_errors=True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
