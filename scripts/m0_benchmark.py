"""M0 Mac benchmark: 256^3 volume, 41 projections, LEAP CPU.

Each case runs in a fresh subprocess so peak RSS (ru_maxrss) is per case.
Iterative time per iteration = (t(n2) - t(n1)) / (n2 - n1), which removes the one-off
setup cost (LEAP computes P1 and the sensitivity P*1 before iterating).
Effective parallelism = (user + sys CPU time) / wall time over the timed call.

Usage: python scripts/m0_benchmark.py
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "geometry_placeholder.yaml"
N = 256
VOXEL = 0.46  # mm, cubic (LEAP requires cubic voxels for this geometry); config voxel_xy
BINNING = 4  # 768 x 768, 0.556 mm
Z_MIN = 20.0
OS_SUBSETS = 8
CASES = [
    ("project", {}),
    ("backproject", {}),
    ("ramp+LEAP BP", {}),
    ("LEAP FBP", {}),
    ("OS-SART", {"n_iter": 1, "subsets": OS_SUBSETS}),
    ("OS-SART", {"n_iter": 3, "subsets": OS_SUBSETS}),
    ("OS-SART LEAP", {"n_iter": 1, "subsets": OS_SUBSETS}),
    ("OS-SART LEAP", {"n_iter": 3, "subsets": OS_SUBSETS}),
    ("SIRT", {"n_iter": 1, "subsets": 1}),
    ("SIRT", {"n_iter": 3, "subsets": 1}),
]


def _case(name: str, kw: dict) -> dict:
    import numpy as np

    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.geometry.loader import load_geometry
    from dtsrecon.geometry.schema import VolumeSpec
    from dtsrecon.phantoms.geometric import sphere_with_beads
    from dtsrecon.recon.simple_fbp import simple_fbp

    geom = load_geometry(CONFIG, binning=BINNING)
    vol = VolumeSpec.from_z_range(N, N, N, VOXEL, Z_MIN, VOXEL)
    zc = vol.center[2]
    ph = sphere_with_beads(vol, 50.0, 1.5, [(-30, 0, zc - 30), (0, 0, zc), (30, 0, zc + 30)])
    eng = LeapEngine(geom, vol)
    g = eng.project(ph.voxelize(vol, supersample=1))

    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    err = None
    try:
        if name == "project":
            eng.project(np.zeros(vol.shape, np.float32))
        elif name == "backproject":
            eng.backproject(g)
        elif name == "ramp+LEAP BP":
            simple_fbp(eng, g)
        elif name == "LEAP FBP":
            eng.fbp(g)
        elif name == "OS-SART LEAP":  # LEAP's own loop: recomputes subset sensitivities every iteration
            eng.sart(g, kw["n_iter"], kw["subsets"], cache_sensitivity=False)
        else:
            eng.sart(g, kw["n_iter"], kw["subsets"])
    except Exception as exc:
        err = repr(exc)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return {
        "case": name, **kw, "wall_s": wall, "cpu_s": cpu, "parallelism": cpu / wall if wall > 0 else None,
        "peak_rss_gb": r1.ru_maxrss / 1e9,  # bytes on macOS
        "error": err,
        "detector": [geom.n_rows, geom.n_cols], "n_proj": geom.n_projections, "volume": list(vol.shape),
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        print(json.dumps(_case(sys.argv[2], json.loads(sys.argv[3]))))
        return
    results = []
    for name, kw in CASES:
        p = subprocess.run([sys.executable, __file__, "--case", name, json.dumps(kw)], capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        r = json.loads(line[-1]) if line else {"case": name, **kw, "error": p.stderr[-2000:]}
        results.append(r)
        print(f"{name:14s} {json.dumps(kw):32s} wall {r.get('wall_s', float('nan')):8.2f} s  cpu/wall {r.get('parallelism') or float('nan'):5.1f}  "
              f"peak {r.get('peak_rss_gb', float('nan')):5.2f} GB  {r.get('error') or ''}", flush=True)

    def per_iter(name):
        a = {r["n_iter"]: r["wall_s"] for r in results if r["case"] == name and not r.get("error")}
        return (a[3] - a[1]) / 2 if 1 in a and 3 in a else None

    summary = {
        "os_sart_subsets": OS_SUBSETS,
        "os_sart_s_per_iter": per_iter("OS-SART"),
        "os_sart_leap_uncached_s_per_iter": per_iter("OS-SART LEAP"),
        "sirt_s_per_iter": per_iter("SIRT"),
        "omp_get_num_procs_expected": os.cpu_count(),
        "machine": platform.platform(),
        "python": platform.python_version(),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "reports" / "m0_benchmark_results.json").write_text(json.dumps({"cases": results, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
