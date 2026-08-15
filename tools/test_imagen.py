"""Imagen generation: availability, prompt building, request shape, caching."""
import base64
import json
import sys
import tempfile
from collections import Counter
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
    # The only engine is Vertex (needs a project); with none configured,
    # generation isn't available.
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

    # Pin Vertex to a single Gemini model + region so this exercises the
    # generateContent path deterministically.
    cfg = {"vertex_project": "proj", "vertex_service_account": "",
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

    # Vertex unconfigured (no project) -> GenError, so the scene falls back to search
    try:
        im.image("x", {}, Path(tempfile.mkdtemp()) / "g3.png")
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
    check("pool defaults to the Gemini image models (Imagen was retired)",
          im._vertex_combos({})[0][0].startswith("gemini-"))
    check("pool defaults to MULTIPLE models (separate quota pools)",
          len({m for m, _ in im._vertex_combos({})}) >= 3)
    check("default regions include global (for global-only models)",
          any(r == "global" for _, r in im._vertex_combos({})))
    check("default region pool is wide (14+)",
          len({r for _, r in im._vertex_combos({})}) >= 14)

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
    vpool = {"vertex_project": "p", "vertex_models": "m", "vertex_regions": "r1\nr2"}
    try:
        im._vertex_reset()
        im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp1.png")
        check("busy combo m@r1 → rotated to m@r2", vseen, [("m", "r1"), ("m", "r2")])
        vseen.clear()
        im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp2.png")
        check("a rested combo is skipped next time", vseen, [("m", "r2")])
        # whole pool busy → raises _RateLimited (WAITABLE), NOT a fatal GenError,
        # so image()'s backoff waits for a combo to free up and retries.
        im._vertex_reset()
        im._vertex_one = lambda *a: (_ for _ in ()).throw(im._VertexBusy())
        try:
            im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp3.png")
            check("exhausted vertex pool raises", False)
        except im._RateLimited as rl:
            check("whole vertex pool busy → _RateLimited (waits, doesn't fail)", True)
            check("carries a retry_after so the scene waits", rl.retry_after is not None)
        except im.GenError:
            check("exhausted pool must WAIT, not GenError", False)
        # a fully-RESTING pool also waits (raises _RateLimited), never fails
        im._vertex_put_to_rest(("m", "r1"), 5)
        im._vertex_put_to_rest(("m", "r2"), 5)
        try:
            im._vertex_raw("x", vpool, Path(tempfile.mkdtemp()) / "vp4.png")
            check("resting pool raises", False)
        except im._RateLimited:
            check("fully-resting pool → _RateLimited (waits, doesn't fail)", True)
        except im.GenError:
            check("resting pool must WAIT, not GenError", False)
        # LOAD-BALANCING: successive calls START on DIFFERENT combos (round-robin),
        # not always the first — so a healthy pool spreads work across every combo.
        im._vertex_one = saved_vone
        im._vertex_reset()
        starts = []
        big = {"vertex_project": "p", "vertex_models": "m",
               "vertex_regions": "r1\nr2\nr3\nr4"}

        def rec_one(project, region, model, sa, prompt, cfg2, dst):
            starts.append((model, region))
            Path(dst).write_bytes(b"VX")          # succeeds immediately (no failover)
        im._vertex_one = rec_one
        for k in range(8):
            im._vertex_raw("x", big, Path(tempfile.mkdtemp()) / f"rr{k}.png")
        check("8 healthy calls hit all 4 regions, evenly (2 each)",
              sorted(Counter(r for _, r in starts).values()), [2, 2, 2, 2])
        check("consecutive calls start on different combos (round-robin)",
              starts[0] != starts[1] and starts[1] != starts[2], True)
    finally:
        im._vertex_one = saved_vone
        im._vertex_reset()

    print("\n  Stop is honoured promptly:")
    im._THROTTLE.reset()
    im._generate = lambda u, b, t: {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": tiny}}]}}]}
    try:
        im.image("stop me", {"vertex_project": "proj",
                             "vertex_models": "gemini-2.5-flash-image",
                             "generate_min_interval": 30},
                 Path(tempfile.mkdtemp()) / "c.png", should_cancel=lambda: True)
        check("a cancelled generation raises Cancelled", False)
    except im.Cancelled:
        check("a cancelled generation raises Cancelled", True)
    except Exception as e:
        check("a cancelled generation raises Cancelled", False, repr(e))

    # ── engine selection (Vertex is the only engine) ────────────────────────
    print("\n  engine selection (Vertex only):")
    im._THROTTLE.reset()
    check("the engine is Vertex", im.selected_engine({}), "vertex")
    check("Vertex needs a project", im.engine_ready("vertex", {}), False)
    check("Vertex ready with a project",
          im.engine_ready("vertex", {"vertex_project": "p"}), True)
    check("order is [vertex] when ready",
          im.engine_order({"vertex_project": "p"}), ["vertex"])
    check("order is empty when Vertex isn't configured", im.engine_order({}), [])
    check("no project → available() False", im.available({}), False)
    check("a project → available() True", im.available({"vertex_project": "p"}), True)

    print("\n  a hard Vertex failure raises GenError (the scene then searches):")
    saved_raw = dict(im._ENGINE_RAW)

    def _boom(prompt, cfg, dest, log=None):
        raise im.GenError("vertex down")
    im._ENGINE_RAW["vertex"] = _boom
    try:
        im.image("x", {"vertex_project": "p", "generate_min_interval": 0,
                       "generate_retries": 0}, Path(tempfile.mkdtemp()) / "f.png")
        check("a dead engine raises GenError (no silent success)", False)
    except im.GenError:
        check("a dead engine raises GenError (no silent success)", True)
    finally:
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw)
        im._THROTTLE.reset()

    print("\n  activity detail: image() surfaces the exact model/region used:")
    im._THROTTLE.reset()

    def _vraw(p, c, d, l=None):
        Path(d).write_bytes(b"IMG")
        return "gemini-2.5-flash-image@us-east4"     # what the pool reports back
    saved_raw3 = dict(im._ENGINE_RAW)
    im._ENGINE_RAW["vertex"] = _vraw
    try:
        det: dict = {}
        im.image("x", {"vertex_project": "p", "generate_min_interval": 0},
                 Path(tempfile.mkdtemp()) / "d.png", detail=det)
        check("detail carries the exact model@region", det.get("model"),
              "gemini-2.5-flash-image@us-east4")
        check("detail label reads 'Vertex · <model>@<region>'", det.get("label"),
              "Vertex · gemini-2.5-flash-image@us-east4")
    finally:
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw3)
        im._THROTTLE.reset()

    print("\n  a busy pool is WAITED OUT and retried, not failed:")
    im._THROTTLE.reset()
    saved_pace2 = im._THROTTLE.pace
    im._THROTTLE.pace = lambda floor, sc=None: None      # don't actually sleep in the test
    saved_raw4 = dict(im._ENGINE_RAW)
    tries = {"n": 0}

    def _busy_then_ok(p, c, d, l=None):
        tries["n"] += 1
        if tries["n"] == 1:
            raise im._RateLimited(retry_after=0.01)      # whole pool busy on the 1st try
        Path(d).write_bytes(b"IMG")
        return "gemini-2.5-flash-image@global"
    im._ENGINE_RAW["vertex"] = _busy_then_ok
    try:
        got = Path(tempfile.mkdtemp()) / "retry.png"
        eng = im.image("x", {"vertex_project": "p", "generate_min_interval": 0,
                             "generate_retries": 3}, got)
        check("a busy pool retries and then succeeds (scene not killed)", eng, "vertex")
        check("it retried exactly once then wrote the image",
              got.exists() and tries["n"] == 2, True)
    finally:
        im._THROTTLE.pace = saved_pace2
        im._ENGINE_RAW.clear()
        im._ENGINE_RAW.update(saved_raw4)
        im._THROTTLE.reset()

    print("\n  generate_workers actually speeds up (pace = floor / workers):")
    im._THROTTLE.reset()
    paces = []
    saved_pace = im._THROTTLE.pace
    im._THROTTLE.pace = lambda floor, sc=None: paces.append(floor)
    saved_raw2 = dict(im._ENGINE_RAW)
    im._ENGINE_RAW["vertex"] = lambda p, c, d, l=None: Path(d).write_bytes(b"IMG")
    wcfg = {"vertex_project": "p"}
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

    # ── manual per-scene generation (review page path) ──────────────────────
    print("\n  manual generate_scenes (the review-page 'Generate' button):")
    from types import SimpleNamespace
    import shutil
    from lib import pipeline as pl

    im.available = lambda cfg: True
    made: list = []

    def fake_image(prompt, cfg, dest, log=None, should_cancel=None, detail=None):
        if "boom" in prompt:
            raise im.GenError("safety filter")     # e.g. a real person
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"IMG")
        made.append(str(dest))
        if isinstance(detail, dict):               # report exactly what made it
            detail.update(engine="vertex", model="gemini-2.5-flash-image@us-east4",
                          label="Vertex · gemini-2.5-flash-image@us-east4")
        return "vertex"                            # image() reports the engine used
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
              res["assets"][1]["src"], "vertex")
        check("the asset records the exact model that made it",
              res["assets"][1]["model"], "gemini-2.5-flash-image@us-east4")
        check("the credit names the engine AND the model",
              res["assets"][1]["credit"],
              "AI-generated (Vertex · gemini-2.5-flash-image@us-east4)")
        prev = res["assets"][1]["path"]
        res2 = pl.generate_scenes(scenes, sheet, {"vertex_project": "p"}, [1],
                                  log=lambda *_: None)
        check("re-generating makes a NEW take (fresh file)",
              res2["assets"][1]["path"] != prev, True)

        # ── concurrency: generate_workers images generate AT ONCE ───────────
        print("\n  concurrent generation (generate_workers images at once):")
        import threading as _thr
        import time as _t
        cur = {"now": 0, "max": 0}
        clock = _thr.Lock()

        def busy_image(prompt, cfg, dest, log=None, should_cancel=None, detail=None):
            with clock:                                # count how many run simultaneously
                cur["now"] += 1
                cur["max"] = max(cur["max"], cur["now"])
            _t.sleep(0.15)                             # simulate a slow API call
            with clock:
                cur["now"] -= 1
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"IMG")
            return "vertex"
        im.image = busy_image
        # scenes 11-16 so this doesn't touch scenes 1/2 the later checks rely on
        many = [SimpleNamespace(n=k, query=f"scene {k}", narration="", media="IMAGE")
                for k in range(11, 17)]
        t0 = _t.time()
        resC = pl.generate_scenes(many, sheet,
                                  {"vertex_project": "p", "generate_workers": 4},
                                  list(range(11, 17)), log=lambda *_: None)
        dt = _t.time() - t0
        check("all 6 scenes generated", resC["generated"], [11, 12, 13, 14, 15, 16])
        check("ran concurrently (more than one image at a time)", cur["max"] > 1, True)
        check("6 slow calls beat the sequential time (0.9s) handily", dt < 0.72, True)
        im.image = fake_image                          # restore for later sections

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
