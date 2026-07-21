#!/usr/bin/env python3
"""Install the pipeline's Python packages into whichever venv is running this.

Shared by setup.sh (macOS/Linux) and setup.bat (Windows) so the install logic
lives in one testable place instead of being written twice in two shell
dialects that behave differently.

Run it with the venv's own interpreter:

    .venv/bin/python3      tools/install_deps.py      # macOS / Linux
    .venv\\Scripts\\python.exe tools\\install_deps.py    # Windows

The tricky part is PyTorch. `pip install chatterbox-tts` pulls in whatever
torch PyPI offers, and on Windows that build has no CUDA in it at all — the
install looks perfectly successful and then every generation silently runs on
the CPU, or dies with "no kernel image is available for execution on the
device". So we install the CUDA build explicitly, then check that the GPU can
actually multiply two matrices before declaring victory.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# NVIDIA's Blackwell cards (RTX 50-series, compute capability sm_120) need a
# CUDA 12.8 or newer build. Older wheels compile fine and then refuse to launch
# a single kernel on the card.
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def pip(*args: str, quiet: bool = False) -> int:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    if quiet:
        cmd.insert(4, "--quiet")
    # cwd=ROOT so a bare "." always means this project, however setup was invoked.
    return subprocess.run(cmd, cwd=ROOT).returncode


def say(msg: str = "") -> None:
    print(msg, flush=True)


def have_nvidia_gpu() -> bool:
    """Is there an NVIDIA driver present? Cheap check before a 3 GB download."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                say(f"     found: {line.strip()}")
            return True
    except Exception:
        pass
    return False


def torch_report() -> dict:
    """What torch is actually installed, and can it really use the GPU?"""
    probe = (
        "import json,torch\n"
        "d={'version':torch.__version__,'cuda_build':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device':None,'works':False,"
        "'error':None,'capability':None}\n"
        "try:\n"
        "    if torch.cuda.is_available():\n"
        "        d['device']=torch.cuda.get_device_name(0)\n"
        "        d['capability']='.'.join(map(str,torch.cuda.get_device_capability(0)))\n"
        "        x=torch.randn(64,64,device='cuda')\n"
        "        _=(x@x).sum().item()\n"     # forces a real kernel launch
        "        d['works']=True\n"
        "    elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available():\n"
        "        d['device']='Apple GPU (MPS)'\n"
        "        x=torch.randn(64,64,device='mps'); _=(x@x).sum().item(); d['works']=True\n"
        "except Exception as e:\n"
        "    d['error']=f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(d))\n"
    )
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout.strip().splitlines()
    if out:
        try:
            import json
            return json.loads(out[-1])
        except Exception:
            pass
    # No JSON came back, so torch could not even be imported. Report the last
    # line of the real error rather than a parsing failure of our own making.
    tail = (r.stderr.strip().splitlines() or ["no output"])[-1]
    return {"error": tail, "works": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip the CUDA build even if an NVIDIA card is present")
    a = ap.parse_args()

    say()
    say("  Faceless Studio — installing packages")
    say("  " + "-" * 38)
    say(f"  python  {platform.python_version()} ({sys.executable})")
    say(f"  system  {platform.system()} {platform.machine()}")

    pip("--upgrade", "pip", quiet=True)

    # resemble-perth (Chatterbox's watermarker) still calls
    #     from pkg_resources import resource_filename
    # and setuptools 81 removed pkg_resources. Without this pin the install
    # "succeeds" and Chatterbox then dies with "'NoneType' object is not
    # callable", because perth swallows its own ImportError and leaves the
    # class as None. Pinned before AND after, because Chatterbox's own
    # dependency resolution can drag setuptools forward again.
    say("\n  [1/5] pinning setuptools (perth still needs pkg_resources)")
    pip("setuptools<81", quiet=True)

    # ORDER MATTERS, and it is the opposite of what you would expect.
    #
    # chatterbox-tts 0.1.7 hard-pins torch==2.6.0. If we install the CUDA build
    # first, pip dutifully rips it out and replaces it with the CPU build from
    # PyPI, and everything then runs about twenty times slower with no error
    # message anywhere. So: let Chatterbox install whatever it wants FIRST, then
    # put the CUDA build on top.
    #
    # Overriding the pin is not optional on a recent card. torch 2.6 predates
    # Blackwell (RTX 50-series) support, so no build of it can drive one at all.
    # Chatterbox runs fine on newer torch in practice — the pin is over-strict.
    say("\n  [2/5] installing Chatterbox (the voice engine)")
    say("     this pulls in PyTorch — several GB, leave it running")
    if pip("--upgrade", "chatterbox-tts") != 0:
        say("\n  !! Chatterbox failed to install. Nothing can be voiced until it does.")
        return 1

    want_cuda = not a.cpu_only and platform.system() in ("Windows", "Linux")
    if want_cuda:
        say("\n  [3/5] looking for an NVIDIA GPU")
        if have_nvidia_gpu():
            say("     replacing Chatterbox's CPU-only PyTorch with the CUDA 12.8 build")
            say("     (pip will warn about a version conflict — that is expected")
            say("      and is explained in README section 6)")
            if pip("--force-reinstall", "torch", "torchaudio",
                   "--index-url", CUDA_INDEX) != 0:
                say("\n  !! The CUDA build failed to install.")
                say("     Check your internet connection and run setup again.")
                return 1
        else:
            say("     no NVIDIA driver found — staying on the CPU build.")
            say("     If you do have an NVIDIA card, install its driver from")
            say("     https://www.nvidia.com/drivers , then run setup again.")
            want_cuda = False
    else:
        say("\n  [3/5] skipping CUDA (no NVIDIA GPU expected on this machine)")

    pip("setuptools<81", quiet=True)      # put it back if anything moved it

    # The `faceless` command. --no-deps is essential: this project declares no
    # dependencies precisely so pip cannot re-resolve the environment here and
    # undo the CUDA build we just installed.
    say("\n  [4/5] installing the 'faceless' command")
    # This console entry point only appears on PATH once the .venv is activated,
    # which most people never do. The ./faceless launcher in the repo root needs
    # no activation and is the path we point people at, so a failure here is not
    # a problem — it is a shortcut on top of a shortcut.
    launch = "faceless start" if platform.system() == "Windows" else "./faceless start"
    if pip("-e", ".", "--no-deps", quiet=True) == 0:
        say(f"     installed. Run:  {launch}")
    else:
        say(f"     skipped — no harm done, the launcher still works:  {launch}")

    say("\n  [5/5] checking the install actually works")
    t = torch_report()
    if t.get("error") and not t.get("works"):
        say(f"     !! {t['error']}")
    say(f"     torch          {t.get('version', '?')}")
    say(f"     CUDA in build  {t.get('cuda_build') or 'none (CPU-only build)'}")
    say(f"     GPU visible    {t.get('device') or 'no'}")
    if t.get("capability"):
        say(f"     compute cap    sm_{t['capability'].replace('.', '')}")

    ok = True
    if want_cuda and not t.get("works"):
        ok = False
        say()
        say("  !! PyTorch cannot use your GPU.")
        if not t.get("cuda_build"):
            say("     Something replaced the CUDA build with a CPU-only one —")
            say("     usually chatterbox-tts, which pins torch==2.6.0.")
            say("     Fix it by re-running just the torch install:")
            say(f"       pip install --force-reinstall torch torchaudio "
                f"--index-url {CUDA_INDEX}")
        else:
            say("     The build has CUDA but no kernel would launch. That usually")
            say("     means the card is newer than this build supports. Try the")
            say("     nightly, which tracks new GPUs first:")
            say("       pip install --pre --force-reinstall torch torchaudio \\")
            say("         --index-url https://download.pytorch.org/whl/nightly/cu128")
        if t.get("error"):
            say(f"     reported: {t['error']}")
    elif t.get("works"):
        say(f"     GPU compute    OK — ran a real kernel on {t['device']}")

    # pkg_resources has to survive everything above.
    r = subprocess.run([sys.executable, "-c", "import pkg_resources"],
                       capture_output=True)
    if r.returncode == 0:
        say("     pkg_resources  available (perth will load)")
    else:
        ok = False
        say("     !! pkg_resources missing — run: pip install \"setuptools<81\"")

    say()
    say("  " + "-" * 38)
    launch = "faceless start" if platform.system() == "Windows" else "./faceless start"
    if ok:
        say(f"  Done. Start it with:   {launch}")
    else:
        say("  Finished with warnings — see the !! lines above before voicing.")
    say()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
