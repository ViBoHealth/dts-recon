"""M6: score the end-to-end reconstructions (scripts/m6_end_to_end.sh) against the dental phantom's ground truth
with the M4 dental metrics. Reads data/m6/recon_*/*.tif; writes reports/m6/e2e_metrics.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_study import dental_metrics  # noqa: E402

from dtsrecon.io.export import read_tiff  # noqa: E402
from dtsrecon.phantoms.dental import dental_phantom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
dp = dental_phantom("medium_dog")
rows = []
for tif in sorted((ROOT / "data" / "m6").glob("**/*.tif")):
    f, vol, meta = read_tiff(tif)
    r = {"stack": str(tif.relative_to(ROOT / "data" / "m6")), "geometry": meta.get("preprocess", {}).get("geometry", "?"),
         **{k: round(v, 3) if isinstance(v, float) else v for k, v in dental_metrics(f, vol, dp).items()}}
    rows.append(r)
    print(r, flush=True)
(ROOT / "reports" / "m6").mkdir(parents=True, exist_ok=True)
with open(ROOT / "reports" / "m6" / "e2e_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
