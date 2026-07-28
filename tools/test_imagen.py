"""Imagen generation: availability, prompt building, request shape, caching."""
import base64
import json
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
    # Vertex readiness is a real config check (incl. google-auth); stub it here so
    # this suite tests the delegation, not the env.
    llm.vertex_ready = lambda cfg: bool((cfg or {}).get("vertex_project"))
    # Generation is available out of the box now (Pollinations needs no key), so
    # available() is True even with no Vertex project. Vertex readiness is tested
    # separately via engine_ready().
    check("generation available with no keys (Pollinations)", im.available({}), True)
    check("Vertex engine needs a project",
          im.engine_ready("vertex", {}), False)
    check("Vertex engine ready with a project",
          im.engine_ready("vertex", {"vertex_project": "p"}), True)

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

    # Isolate the Vertex engine for these checks: make the other engines fail
    # instantly so failover can't reach the real network.
    def _no_net(*a, **k):
        raise im.GenError("no network in test")
    im._ENGINE_RAW["pollinations"] = _no_net
    im._ENGINE_RAW["cloudflare"] = _no_net

    cfg = {"vertex_project": "proj", "generate_location": "global",
           "vertex_service_account": "", "generate_engine": "vertex"}
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

    # no working engine (Vertex chosen but unconfigured, others stubbed off) -> GenError
    try:
        im.image("x", {"generate_engine": "vertex"},
                 Path(tempfile.mkdtemp()) / "g3.png")
        check("no working engine raises", False)
    except im.GenError:
        check("no working engine raises GenError", True)

    # ── rate limiting: back off on 429 instead of surrendering ──────────────
    print("\n  429 backoff + throttle:")
    import io
    import urllib.error
    from types import SimpleNamespace as _NS

    check("retryDelay is read from the server's 429 body",
          im._server_retry_after('{"error":{"details":[{"retryDelay":"7s"}]}}', None), 7.5)
    check("Retry-After header is honoured when there's no body delay",
          im._server_retry_after("{}", {"Retry-After": "9"}), 9.5)
    check("no server hint -> None (caller backs off exponentially)",
          im._server_retry_after("{}", None), None)
    check("generate_workers sizes the concurrency gate",
          im._semaphore({"generate_workers": 2})._value, 2)

    def _http_error(code, body=b"{}"):
        return urllib.error.HTTPError("http://x", code, "busy", {}, io.BytesIO(body))

    # Don't actually sleep during the test; record what we'd have waited.
    saved_time, sleeps = im.time, []
    im.time = _NS(sleep=lambda s: sleeps.append(s), monotonic=saved_time.monotonic)
    im._THROTTLE.reset()
    rl_cfg = {"vertex_project": "proj", "generate_location": "global",
              "vertex_service_account": "", "generate_min_interval": 0,
              "generate_retries": 5, "generate_workers": 1,
              "generate_engine": "vertex"}
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
        check("waited out the server's cooldown (~3.5s)",
              any(abs(s - 3.5) < 0.3 for s in sleeps))
        check("a 429 widened the gap for the next scene (adaptive)",
              im._THROTTLE.gap >= 3.5)
        check("surfaced a retry notice to the log", any("retry" in m for m in notes))

        im._THROTTLE.reset()
        im._generate = lambda u, b, t: (_ for _ in ()).throw(_http_error(429))
        try:
            im.image("always busy", {**rl_cfg, "generate_retries": 2},
                     Path(tempfile.mkdtemp()) / "r2.png")
            check("a persistent 429 eventually raises", False)
        except im.GenError as e:
            check("a persistent 429 gives up (all engines failed)",
                  "rate-limited" in str(e) and "engines failed" in str(e))
    finally:
        im.time = saved_time
        im._THROTTLE.reset()

    print("\n  Stop is honoured promptly (no waiting out the backoff):")
    im._THROTTLE.reset()
    im._generate = lambda u, b, t: {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": tiny}}]}}]}          # would succeed if not cancelled
    try:
        im.image("stop me", {"vertex_project": "proj", "generate_location": "global",
                             "vertex_service_account": "", "generate_min_interval": 30},
                 Path(tempfile.mkdtemp()) / "c.png", should_cancel=lambda: True)
        check("a cancelled generation raises Cancelled", False)
    except im.Cancelled:
        check("a cancelled generation raises Cancelled", True)
    except Exception as e:
        check("a cancelled generation raises Cancelled", False, repr(e))

    # ── engine selection + failover chain ───────────────────────────────────
    print("\n  engine selection + failover chain:")
    im._THROTTLE.reset()
    check("default engine is Pollinations", im.selected_engine({}), "pollinations")
    check("Pollinations needs no key", im.engine_ready("pollinations", {}), True)
    check("Cloudflare needs id AND token",
          im.engine_ready("cloudflare", {"cf_account_id": "a"}), False)
    check("Cloudflare ready with id + token",
          im.engine_ready("cloudflare", {"cf_account_id": "a", "cf_api_token": "t"}), True)
    check("order = chosen first, then chain, ready only",
          im.engine_order({"generate_engine": "cloudflare",
                           "cf_account_id": "a", "cf_api_token": "t"}),
          ["cloudflare", "pollinations"])
    check("generation is available out of the box (Pollinations)", im.available({}), True)

    calls = []

    def _bad(name):
        def f(prompt, cfg, dest, log=None):
            calls.append(name)
            raise im.GenError(name + " down")
        return f

    def _good(name):
        def f(prompt, cfg, dest, log=None):
            calls.append(name)
            Path(dest).write_bytes(b"IMG")
        return f

    saved_raw = dict(im._ENGINE_RAW)
    im._ENGINE_RAW["pollinations"] = _bad("pollinations")
    im._ENGINE_RAW["cloudflare"] = _good("cloudflare")
    im._ENGINE_RAW["vertex"] = _bad("vertex")
    try:
        d = im.image("x", {"cf_account_id": "a", "cf_api_token": "t",
                           "generate_min_interval": 0, "pollinations_interval": 0},
                     Path(tempfile.mkdtemp()) / "fo.png")
        check("failover: Pollinations failed → Cloudflare made it",
              calls, ["pollinations", "cloudflare"])
        check("failover wrote the image", d.exists() and d.stat().st_size > 0)
    finally:
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw)
        im._THROTTLE.reset()

    # ── Cloudflare multi-account pooling + per-account cooldown ──────────────
    print("\n  Cloudflare multi-account rotation:")
    im._cf_reset()
    check("parses the single pair",
          im._cloudflare_accounts({"cf_account_id": "a1", "cf_api_token": "t1"}),
          [("a1", "t1")])
    check("parses a pool (id:token per line) and dedups by id",
          im._cloudflare_accounts({"cf_account_id": "a1", "cf_api_token": "t1",
                                   "cf_accounts": "a2:t2\na3 t3\na1:tX"}),
          [("a1", "t1"), ("a2", "t2"), ("a3", "t3")])
    check("Cloudflare is ready with only a pool",
          im.engine_ready("cloudflare", {"cf_accounts": "a2:t2"}), True)
    check("cap detection: 429 is a cap", im._cf_is_cap(429, ""), True)
    check("cap detection: neuron-limit 400 is a cap",
          im._cf_is_cap(400, "free neuron limit exceeded"), True)
    check("cap detection: a content 400 is NOT a cap",
          im._cf_is_cap(400, "Input prompt contains inappropriate content"), False)

    print("\n  Cloudflare model-aware request body (quality params):")
    sdxl = im._cf_body("@cf/bytedance/stable-diffusion-xl-lightning", "a cat",
                       {"cf_steps": 20, "generate_aspect": "16:9"})
    check("SDXL body has 16:9 width/height", (sdxl["width"], sdxl["height"]), (1280, 720))
    check("SDXL body sends num_steps", sdxl["num_steps"], 20)
    check("SDXL body sends a negative prompt (pushes realism)",
          bool(sdxl.get("negative_prompt")))
    flux = im._cf_body("@cf/black-forest-labs/flux-1-schnell", "a cat", {"cf_steps": 20})
    check("flux body caps steps at 8", flux["steps"], 8)
    check("flux body has no width/height (fixed square)", "width" not in flux)
    check("flux body has no negative prompt", "negative_prompt" not in flux)

    seen = []
    saved_one = im._cf_one

    def cf_one(aid, tok, model, prompt, dest, cfg):
        seen.append(aid)
        if aid == "a1":
            raise im._CFRateLimited()          # first account is over its cap
        Path(dest).write_bytes(b"CF")
    im._cf_one = cf_one
    cf_cfg = {"cf_accounts": "a1:t1\na2:t2", "cf_rest_minutes": 15}
    try:
        im._cf_reset()
        im._cloudflare_raw("x", cf_cfg, Path(tempfile.mkdtemp()) / "c1.png")
        check("rotated a1 (capped) → a2 (made the image)", seen, ["a1", "a2"])
        seen.clear()
        im._cloudflare_raw("x", cf_cfg, Path(tempfile.mkdtemp()) / "c2.png")
        check("a rested account is skipped next time", seen, ["a2"])

        im._cf_reset()

        def cf_all(aid, tok, model, prompt, dest, cfg):
            raise im._CFRateLimited()
        im._cf_one = cf_all
        try:
            im._cloudflare_raw("x", cf_cfg, Path(tempfile.mkdtemp()) / "c3.png")
            check("an exhausted pool raises", False)
        except im.GenError as e:
            check("whole pool exhausted → GenError (fast engine failover)",
                  "rate-limited" in str(e))
    finally:
        im._cf_one = saved_one
        im._cf_reset()

    # ── manual per-scene generation (review page path) ──────────────────────
    print("\n  manual generate_scenes (the review-page 'Generate' button):")
    from types import SimpleNamespace
    import shutil
    from lib import pipeline as pl

    im.available = lambda cfg: True
    made: list = []

    def fake_image(prompt, cfg, dest, log=None, should_cancel=None):
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

        # ── user upload (review 'Upload your own') ──────────────────────────
        print("\n  upload your own image / video (only the target scene changes):")
        keep1 = res2["assets"][1]["path"]
        up = pl.set_scene_upload(sheet, 2, b"MYPHOTO", "png")
        assets = json.loads(pl.paths_for(sheet, "en")["assets"].read_text())
        check("scene 2 now points at the uploaded file", assets["2"]["src"], "upload")
        check("uploaded image media is IMAGE", up["media"], "IMAGE")
        check("the uploaded bytes were written",
              (pl.paths_for(sheet, "en")["stockcache"] / up["file"]).read_bytes(), b"MYPHOTO")
        check("scene 1 was left untouched", assets["1"]["path"], keep1)
        vid = pl.set_scene_upload(sheet, 1, b"MYCLIP", "MP4")
        check("uploaded .mp4 is tagged VIDEO", vid["media"], "VIDEO")
        check("upload filename is project-namespaced", vid["file"].startswith("up_"), True)
        try:
            pl.set_scene_upload(sheet, 1, b"x", "txt")
            check("a bad type is rejected", False)
        except ValueError:
            check("a bad file type raises ValueError", True)
        try:
            pl.set_scene_upload(sheet, 1, b"", "png")
            check("an empty file is rejected", False)
        except ValueError:
            check("an empty file raises ValueError", True)
    finally:
        shutil.rmtree(root / "projects" / tpid, ignore_errors=True)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURE(S)'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
