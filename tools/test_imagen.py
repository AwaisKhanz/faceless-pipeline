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
    # Engines are Cloudflare (needs keys) and Vertex (needs a project); with
    # neither configured, generation isn't available.
    check("no engine configured -> unavailable", im.available({}), False)
    check("available with a Vertex project", im.available({"vertex_project": "p"}), True)
    check("Vertex engine needs a project", im.engine_ready("vertex", {}), False)
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

    # Isolate Vertex: pin it to a single Gemini model + region and turn failover
    # off, so image() uses ONLY the vertex generateContent path here.
    cfg = {"vertex_project": "proj", "vertex_service_account": "",
           "generate_engine": "vertex", "generate_failover": False,
           "vertex_models": "gemini-2.5-flash-image", "vertex_regions": "us-central1"}
    dest = Path(tempfile.mkdtemp()) / "g.png"
    eng = im.image("a rocket launch", cfg, dest)
    check("writes the image file", dest.exists() and dest.stat().st_size > 0)
    check("reports the engine that produced it", eng, "vertex")
    check("endpoint targets the project + region",
          "projects/proj/locations/us-central1" in cap["url"])
    check("a Gemini image model uses generateContent",
          cap["url"].endswith(":generateContent"))
    check("endpoint targets the chosen model", "gemini-2.5-flash-image" in cap["url"])
    check("asks for an IMAGE modality",
          "IMAGE" in cap["body"]["generationConfig"]["responseModalities"])
    check("asks for 16:9",
          cap["body"]["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")
    check("passes the bearer token", cap["token"], "TOK")

    # caching: a second call for the same dest must NOT hit the API
    def boom(*a, **k):
        raise AssertionError("should have used the cached file")
    im._generate = boom
    im.image("a rocket launch", cfg, dest)     # must NOT call _generate (boom)
    check("a re-run reuses the cached file, no second call", dest.exists())

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

    # ── helpers + Vertex model×region pool ──────────────────────────────────
    print("\n  helpers + Vertex pool:")
    check("retryDelay is read from the server's 429 body",
          im._server_retry_after('{"error":{"details":[{"retryDelay":"7s"}]}}', None), 7.5)
    check("Retry-After honoured when no body delay",
          im._server_retry_after("{}", {"Retry-After": "9"}), 9.5)
    check("no server hint -> None", im._server_retry_after("{}", None), None)
    check("generate_workers sizes the concurrency gate",
          im._semaphore({"generate_workers": 2})._value, 2)

    check("pool = models × regions (deduped)",
          im._vertex_combos({"vertex_models": "m1, m2", "vertex_regions": "r1\nr2"}),
          [("m1", "r1"), ("m1", "r2"), ("m2", "r1"), ("m2", "r2")])
    check("pool defaults to the Imagen models",
          im._vertex_combos({})[0][0].startswith("imagen-"))

    # Imagen models use :predict (dedicated image model, its own quota).
    cap2 = {}
    im._generate = lambda url, body, token: (
        cap2.update(url=url, body=body)
        or {"predictions": [{"bytesBase64Encoded": tiny, "mimeType": "image/png"}]})
    im._vertex_reset()
    pdest = Path(tempfile.mkdtemp()) / "imagen.png"
    im._vertex_raw("a cat", {"vertex_project": "proj", "vertex_service_account": "",
                             "vertex_models": "imagen-3.0-generate-002",
                             "vertex_regions": "us-central1"}, pdest)
    check("Imagen uses the :predict endpoint", cap2["url"].endswith(":predict"))
    check("Imagen endpoint targets the region", "locations/us-central1" in cap2["url"])
    check("Imagen sends instances[].prompt", cap2["body"]["instances"][0]["prompt"], "a cat")
    check("Imagen writes the image", pdest.exists() and pdest.stat().st_size > 0)

    # rotation: a busy (model, region) is rested; another combo carries the load.
    vseen = []
    saved_vone = im._vertex_one

    def v_one(project, region, model, sa, prompt, cfg2, dst):
        vseen.append((model, region))
        if region == "r1":
            raise im._VertexBusy()
        Path(dst).write_bytes(b"VX")
    im._vertex_one = v_one
    vpool = {"vertex_project": "p", "vertex_models": "m",
             "vertex_regions": "r1\nr2", "vertex_rest_minutes": 5}
    try:
        im._vertex_reset()
        im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp1.png")
        check("busy combo m@r1 → rotated to m@r2", vseen, [("m", "r1"), ("m", "r2")])
        vseen.clear()
        im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp2.png")
        check("a rested combo is skipped next time", vseen, [("m", "r2")])
        im._vertex_reset()
        im._vertex_one = lambda *a: (_ for _ in ()).throw(im._VertexBusy())
        try:
            im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp3.png")
            check("exhausted vertex pool raises", False)
        except im.GenError as e:
            check("whole vertex pool busy → GenError", "busy" in str(e))
    finally:
        im._vertex_one = saved_vone
        im._vertex_reset()

    print("\n  Stop is honoured promptly:")
    im._THROTTLE.reset()
    im._generate = lambda u, b, t: {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": tiny}}]}}]}
    try:
        im.image("stop me", {"vertex_project": "proj", "generate_engine": "vertex",
                             "vertex_models": "gemini-2.5-flash-image",
                             "generate_min_interval": 30},
                 Path(tempfile.mkdtemp()) / "c.png", should_cancel=lambda: True)
        check("a cancelled generation raises Cancelled", False)
    except im.Cancelled:
        check("a cancelled generation raises Cancelled", True)
    except Exception as e:
        check("a cancelled generation raises Cancelled", False, repr(e))

    # ── engine selection + failover chain ───────────────────────────────────
    print("\n  engine selection + failover chain:")
    im._THROTTLE.reset()
    check("default engine is Cloudflare", im.selected_engine({}), "cloudflare")
    check("Cloudflare needs id AND token",
          im.engine_ready("cloudflare", {"cf_account_id": "a"}), False)
    check("Cloudflare ready with id + token",
          im.engine_ready("cloudflare", {"cf_account_id": "a", "cf_api_token": "t"}), True)
    check("order = chosen first, then chain, ready only",
          im.engine_order({"generate_engine": "vertex", "vertex_project": "p",
                           "cf_account_id": "a", "cf_api_token": "t"}),
          ["vertex", "cloudflare"])
    check("failover OFF → ONLY the chosen engine (no switching)",
          im.engine_order({"generate_engine": "vertex", "generate_failover": False,
                           "vertex_project": "p"}),
          ["vertex"])
    check("no engine configured → available() False", im.available({}), False)

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
    im._ENGINE_RAW["cloudflare"] = _bad("cloudflare")
    im._ENGINE_RAW["vertex"] = _good("vertex")
    try:
        fo = Path(tempfile.mkdtemp()) / "fo.png"
        eng = im.image("x", {"generate_engine": "cloudflare", "cf_account_id": "a",
                             "cf_api_token": "t", "vertex_project": "p",
                             "generate_min_interval": 0}, fo)
        check("failover: Cloudflare failed → Vertex made it",
              calls, ["cloudflare", "vertex"])
        check("failover wrote the image", fo.exists() and fo.stat().st_size > 0)
        check("reports the engine that actually succeeded", eng, "vertex")
    finally:
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw)
        im._THROTTLE.reset()

    print("\n  generate_workers actually speeds up (pace = floor / workers):")
    im._THROTTLE.reset()
    paces = []
    saved_pace = im._THROTTLE.pace
    im._THROTTLE.pace = lambda floor, sc=None: paces.append(floor)
    saved_raw2 = dict(im._ENGINE_RAW)
    im._ENGINE_RAW["cloudflare"] = lambda p, c, d, l=None: Path(d).write_bytes(b"IMG")
    wcfg = {"generate_engine": "cloudflare", "cf_account_id": "a", "cf_api_token": "t"}
    try:
        im.image("x", {**wcfg, "generate_min_interval": 8, "generate_workers": 4},
                 Path(tempfile.mkdtemp()) / "w4.png")
        check("4 workers → pace floor is base/4 (8→2s)", abs(paces[-1] - 2.0) < 0.01)
        paces.clear()
        im.image("y", {**wcfg, "generate_min_interval": 8, "generate_workers": 1},
                 Path(tempfile.mkdtemp()) / "w1.png")
        check("1 worker → full pace floor (8s)", abs(paces[-1] - 8.0) < 0.01)
    finally:
        im._THROTTLE.pace = saved_pace
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw2)
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

    print("\n  Cloudflare model-aware request (quality params + FLUX.2 multipart):")
    sdxl_data, sdxl_ct = im._cf_request("@cf/bytedance/stable-diffusion-xl-lightning",
                                        "a cat", {"cf_steps": 20, "generate_aspect": "16:9"})
    sdxl = json.loads(sdxl_data)
    check("SDXL uses JSON", sdxl_ct, "application/json")
    check("SDXL has 16:9 width/height", (sdxl["width"], sdxl["height"]), (1280, 720))
    check("SDXL sends num_steps", sdxl["num_steps"], 20)
    check("SDXL sends a negative prompt (pushes realism)", bool(sdxl.get("negative_prompt")))
    flux = json.loads(im._cf_request("@cf/black-forest-labs/flux-1-schnell",
                                     "a cat", {"cf_steps": 20})[0])
    check("flux-1 caps steps at 8", flux["steps"], 8)
    check("flux-1 has no width/height (fixed square)", "width" not in flux)
    f2_data, f2_ct = im._cf_request("@cf/black-forest-labs/flux-2-dev", "a cat",
                                    {"cf_steps": 28, "generate_aspect": "16:9"})
    check("FLUX.2 uses multipart form-data (not JSON)",
          f2_ct.startswith("multipart/form-data"))
    check("FLUX.2 form carries prompt + 16:9 + 28 steps",
          b'name="prompt"' in f2_data and b'name="width"' in f2_data
          and b"1280" in f2_data and b"28" in f2_data)
    k4_data, k4_ct = im._cf_request("@cf/black-forest-labs/flux-2-klein-4b", "a cat",
                                    {"generate_aspect": "16:9"})
    check("FLUX.2 klein also uses multipart", k4_ct.startswith("multipart/form-data"))
    check("klein defaults to few steps (distilled)", b'name="steps"' in k4_data and b"\r\n\r\n8\r\n" in k4_data)
    check("extract top-level image (FLUX.2 shape)",
          im._cf_extract_b64(b'{"image":"AAA"}'), "AAA")
    check("extract result.image (flux-1 shape)",
          im._cf_extract_b64(b'{"result":{"image":"BBB"}}'), "BBB")

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
        return "cloudflare"                        # image() reports the engine used
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
        check("the asset's source is the engine that made it (not always 'imagen')",
              res["assets"][1]["src"], "cloudflare")
        check("the credit names the engine", res["assets"][1]["credit"], "AI-generated (Cloudflare)")
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
