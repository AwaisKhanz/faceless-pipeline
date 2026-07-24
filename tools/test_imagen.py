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
    check("no project -> unavailable", im.available({}), False)
    check("a project -> available", im.available({"vertex_project": "p"}), True)

    p = im.prompt_for("data center servers", {})
    check("prompt keeps the subject", "data center servers" in p)
    check("prompt adds a consistent style", "cinematic" in p)
    check("a custom style overrides the default",
          "neon" in im.prompt_for("x", {"generate_style": "neon"}))

    # image(): mock the token and the HTTP predict call
    llm._vertex_token = lambda sa: "TOK"
    cap = {}
    tiny = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode()

    def fake_predict(url, body, token):
        cap.update(url=url, body=body, token=token)
        return {"predictions": [{"bytesBase64Encoded": tiny, "mimeType": "image/png"}]}
    im._predict = fake_predict

    cfg = {"vertex_project": "proj", "generate_location": "us-central1",
           "vertex_service_account": ""}
    dest = Path(tempfile.mkdtemp()) / "g.png"
    out = im.image("a rocket launch", cfg, dest)
    check("writes the image file", out.exists() and out.stat().st_size > 0)
    check("endpoint targets project + region",
          "projects/proj/locations/us-central1" in cap["url"])
    check("endpoint targets the default model",
          "imagen-3.0-generate-002:predict" in cap["url"])
    check("asks for exactly ONE image", cap["body"]["parameters"]["sampleCount"], 1)
    check("asks for 16:9", cap["body"]["parameters"]["aspectRatio"], "16:9")
    check("passes the bearer token", cap["token"], "TOK")

    # caching: a second call for the same dest must NOT hit the API
    def boom(*a, **k):
        raise AssertionError("should have used the cached file")
    im._predict = boom
    out2 = im.image("a rocket launch", cfg, dest)
    check("a re-run reuses the cached file, no second call", out2 == dest)

    # an empty prediction (safety filter) -> GenError, not a broken file
    im._predict = lambda url, body, token: {"predictions": [{"raiFilteredReason": "blocked"}]}
    try:
        im.image("x", cfg, Path(tempfile.mkdtemp()) / "g2.png")
        check("empty prediction raises", False)
    except im.GenError:
        check("empty prediction raises GenError", True)

    # missing project -> GenError before any network call
    try:
        im.image("x", {"generate_location": "us-central1"},
                 Path(tempfile.mkdtemp()) / "g3.png")
        check("missing project raises", False)
    except im.GenError:
        check("missing project raises GenError", True)

    # ── manual per-scene generation (review page path) ──────────────────────
    print("\n  manual generate_scenes (the review-page 'Generate' button):")
    from types import SimpleNamespace
    import shutil
    from lib import pipeline as pl

    im.available = lambda cfg: True
    made: list = []

    def fake_image(prompt, cfg, dest):
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
