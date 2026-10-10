# dts-recon

Digital tomosynthesis reconstruction for the NextGen PANEL moving-source prototype (2D source scan, stationary detector; first use case veterinary dental).
Project brief: `docs/PROJECT_BRIEF.md`. Working rules: `CLAUDE.md`. Status: Pass A (simulation), M0.

## Mac setup (Apple Silicon, CPU only)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,leap]' cmake
brew install libomp
third_party/leap/build_leap_mac.sh /path/outside/repo/leap_build   # LEAP v1.26 + CPU build patch
.venv/bin/pytest
```

## Scripts

- `scripts/smoke_grid.py` — device geometry (`configs/geometry_dental.yaml`): sphere + beads, depth check.
- `scripts/m0_smoke.py` — 64³ sphere + beads, LEAP projection, ramp+BP and OS-SART, bead depth check.
- `scripts/m0_benchmark.py` — 256³ / 41-projection timing and peak memory.
- `dts plot-geometry configs/geometry_dental.yaml [-o out.png]`

All geometry values in `configs/` are placeholders.
