#!/usr/bin/env python3
"""Why is perth.PerthImplicitWatermarker None?

Chatterbox fails with "'NoneType' object is not callable" when the watermarker
class is missing. perth swallows its own import error, so this reproduces the
import directly and shows the real cause.
"""
import importlib
import sys
import traceback

print("python:", sys.version.split()[0])
print()

try:
    import perth
except Exception:
    print("perth will not import at all:")
    traceback.print_exc()
    raise SystemExit(1)

print("perth package:", getattr(perth, "__file__", "?"))
print("perth version:", getattr(perth, "__version__", "unknown"))
attr = getattr(perth, "PerthImplicitWatermarker", "MISSING")
print("PerthImplicitWatermarker:", attr)
print()

if attr not in (None, "MISSING"):
    print("It is present — the problem is elsewhere.")
    raise SystemExit(0)

print("Reproducing the swallowed import, one submodule at a time:")
candidates = [
    "perth.perth_net",
    "perth.perth_net.perth_net_implicit",
    "perth.perth_net.perth_net_implicit.perth_watermarker",
]
for name in candidates:
    try:
        importlib.import_module(name)
        print(f"  ok   {name}")
    except Exception as e:
        print(f"  FAIL {name}")
        print(f"       {type(e).__name__}: {e}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        break

print()
print("Dependency versions that commonly cause this:")
for mod in ("setuptools", "pkg_resources", "numpy", "librosa", "torch", "resampy"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"  {mod:<14} MISSING ({type(e).__name__})")
