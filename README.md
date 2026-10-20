# dts-recon

Digital tomosynthesis reconstruction for the NextGen PANEL moving-source prototype (2D source scan, stationary detector; first use case veterinary dental).
Project brief: `docs/PROJECT_BRIEF.md`. Working rules: `CLAUDE.md`. Status: Pass A (simulation), M0–M6 built; first real-data test on an older system (`reports/real_data_v1.md`); see `docs/STATUS.md`.

## Mac setup (Apple Silicon, CPU only)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,leap,dicom,viz]' cmake
brew install libomp
third_party/leap/build_leap_mac.sh /path/outside/repo/leap_build   # LEAP v1.26 + CPU build patch
.venv/bin/pytest
```

## Using the tool

`docs/USAGE.md`: mock raw scan -> calibration -> reconstruction -> TIFF / DICOM -> napari. In short:

```bash
.venv/bin/dts mock-scan data/dog_scan --phantom dental
.venv/bin/dts recon data/dog_scan -o data/recon --method os-sart
.venv/bin/dts view data/recon/os-sart.tif
```

Other commands: `dts info`, `dts calibrate`, `dts plot-geometry configs/geometry_dental.yaml [-o out.png]`.

## Studies (scripts/)

- `m0_smoke.py`, `m0_benchmark.py`: linear-sweep smoke test and Mac timing / peak memory (M0).
- `smoke_grid.py`, `sim_dental.py`: device geometry smoke test; dental phantom through S0/S1/S2.
- `m3_compare.py`, `m3_dental_lesions.py`: bead and lesion depths for the three methods (M3).
- `m4_study.py`, `m4_plots.py`, `m4_gaps.py`, `m4_memory.py`: trade study, motion / slice-thickness / truncation, compute (M4).
- `m5_study.py`, `m5_plots.py`: calibration accuracy, geometry-error sensitivity, tolerance table (M5).
- `m6_end_to_end.sh`, `m6_check.py`: the full tool chain on mock data and its scoring (M6).
- `export_stack.py`: dental slice stacks for all methods plus a self-contained HTML viewer.

Results and decisions: `docs/decisions.md`. All physics, detector and geometry values are placeholders.
