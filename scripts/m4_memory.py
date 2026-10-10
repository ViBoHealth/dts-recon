"""M4 compute check (brief §2 / §11): wall time and peak resident memory per method on the dental phantom,
6x6 / 400 / SID 500, at two binnings (8x: 0.8 mm voxels, the M4 study size; 4x: 0.4 mm). Each case runs in a
fresh process so ru_maxrss is that case's peak. CPU (LEAP), Mac.

Usage: python scripts/m4_memory.py            -> runs all cases, writes reports/m4/compute.csv
       python scripts/m4_memory.py BIN METHOD -> one case (internal)
"""

from __future__ import annotations

import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [(b, m) for b in (8, 4) for m in ("DTS-FBP", "OS-SART", "ASD-POCS")]


def one(bin_: int, method: str):
    import numpy as np

    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.geometry.generators import grid_trajectory
    from dtsrecon.geometry.schema import VolumeSpec
    from dtsrecon.phantoms.dental import dental_phantom
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import ASD_POCS_TUNED, asd_pocs, os_sart
    from dtsrecon.simulate.acquisition import StageConfig, simulate
    from dtsrecon.simulate.forward import NoiseModel

    p = 0.1 * bin_
    geom = grid_trajectory((6, 6), (400, 400), 500.0, 3000 // bin_, 2000 // bin_, p, p)
    dp = dental_phantom("medium_dog")
    nz = int(round(128 * 0.8 / p))
    vol = VolumeSpec.from_z_range(int(round(300 * 0.8 / p)), int(round(175 * 0.8 / p)), nz, p, dp.z_range[0] - 3 + p / 2, p)
    g = simulate(dp.phantom, geom, vol, LeapEngine, StageConfig(noise=NoiseModel()), np.random.default_rng(0))
    rss_sim = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    eng = LeapEngine(geom, vol)
    t = time.perf_counter()
    if method == "DTS-FBP":
        dts_fbp(eng, g, DTSFBPParams(mode="radial"))
    elif method == "OS-SART":
        os_sart(eng, g, 10, 6)
    else:
        asd_pocs(eng, g, n_iter=10, n_subsets=6, **ASD_POCS_TUNED)
    dt = time.perf_counter() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    n_vox = vol.nx * vol.ny * vol.nz
    print(json.dumps({"bin": bin_, "voxel_mm": p, "volume": f"{vol.nx}x{vol.ny}x{vol.nz}", "volume_gb": round(n_vox * 4 / 2**30, 2),
                      "projections_gb": round(g.size * 4 / 2**30, 2), "method": method, "recon_s": round(dt, 1),
                      "peak_rss_gb": round(rss, 2), "peak_rss_after_sim_gb": round(rss_sim / 2**30, 2)}))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        one(int(sys.argv[1]), sys.argv[2])
        sys.exit()
    rows = []
    for b, m in CASES:
        out = subprocess.run([sys.executable, __file__, str(b), m], capture_output=True, text=True, check=True).stdout
        r = json.loads(out.strip().splitlines()[-1])
        print(r, flush=True)
        rows.append(r)
    with open(ROOT / "reports" / "m4" / "compute.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
