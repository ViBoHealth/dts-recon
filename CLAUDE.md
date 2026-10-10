# CLAUDE.md — dts-recon

Digital tomosynthesis reconstruction for the NextGen PANEL moving-source prototype: 2D source scan over a stationary detector; first use case veterinary dental.
Full context, milestones and acceptance criteria: `docs/PROJECT_BRIEF.md`. Read it before starting work.

## Current phase

Pass A (simulation only). No real data exists yet. All geometry values in `configs/` are **placeholders**.

## Working rules

- **Evidence first.** Report measured numbers and test results. Separate what was measured from what is inferred. When a result is uncertain, say so; don't round it up into a conclusion.
- **Ground truth before claims.** A reconstruction method counts as working only when a simulated phantom with known ground truth shows it.
- **Geometry convention is fixed** (brief §4.1: mm, origin at detector center, +z toward source, +x along the detector column direction). Do not change it without an entry in `docs/decisions.md` and Gil's approval.
- **Engine isolation.** Only `src/dtsrecon/engines/` may import `tigre` or `leapct`.
- **Placeholders stay labeled.** Any value without a confirmed source carries a `# PLACEHOLDER` comment or a `placeholder: true` YAML field.
- **New dependencies:** ask before adding any beyond numpy, scipy, matplotlib, pyyaml, pydantic, tifffile, pydicom, napari, pytest, tigre, leapct, imageio (approved 2026-10-09: hard import of leapct's `leapctype.py`), xraydb (approved 2026-10-09: NIST attenuation data for S2).
- **Milestone reports** go in `reports/Mx_<name>.md`, with figures in `reports/figures/`. Each report states the simulation stages used (S0–S5), the geometry config, and the hardware the run used.

## Machines

- Development happens on an Apple M5 Max laptop (36 GB, no CUDA). LEAP runs here in CPU mode; TIGRE does not.
- TIGRE, GPU tests and full-resolution runs happen on a rented NVIDIA GPU. Say before starting anything that needs it, so the machine is only on when used.

## Testing

- `pytest` runs CPU-only tests by default and must stay fast (< 2 min). Use small volumes (≤ 64³) and few projections.
- GPU tests: `@pytest.mark.gpu`, run with `pytest -m gpu` on the GPU machine only. TIGRE tests are always GPU tests.
- Any change to geometry or engine code must keep the round-trip test passing (< 0.1 px bead centroid agreement against the NumPy reference projector).

## Commands

```
uv pip install --python .venv/bin/python -e '.[dev,leap]'    # Python 3.12 venv; see README for LEAP build
third_party/leap/build_leap_mac.sh <build_dir>               # LEAP v1.26 CPU build (patched) into .venv
pytest                                                        # CPU tests, ~1 s
pytest -m gpu                                                 # GPU machine only
dts plot-geometry configs/geometry_dental.yaml
python scripts/smoke_grid.py                                  # device (2D grid) smoke test
python scripts/sim_dental.py                                  # dental phantom -> S0/S1/S2 -> OS-SART
python scripts/m3_compare.py                                  # M3: bead depths, DTS-FBP / OS-SART / ASD-POCS
python scripts/m3_dental_lesions.py                           # M3: dental lesion depths (S2+S1)
python scripts/m0_smoke.py; python scripts/m0_benchmark.py   # linear-sweep comparison
```

## LEAP gotchas (measured in M0, see docs/decisions.md)

- Our geometry requires cubic voxels in LEAP; the adapter raises on any mismatch.
- LEAP's FBP does not support our geometry; filter in NumPy and use the engine backprojector.
- LEAP CPU must be built with `third_party/leap/joseph_modular_cpu_fix.patch` (projector half-voxel bug + backprojection speedup); the adapter refuses otherwise. Keep the projection/backprojection geometry and adjoint tests passing.
