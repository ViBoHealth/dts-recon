# M0 — Mac setup, LEAP CPU smoke test and benchmark

**Date:** 2026-10-09 · **Scope:** M0, Mac only (no GPU machine, no TIGRE)
**Hardware:** Apple M5 Max (6 performance + 12 efficiency cores = 18), 36 GB unified memory, macOS 27.0.1, on AC power, normal power mode
**Simulation stage:** S0 only (monoenergetic line integrals, no noise)
**Geometry:** `configs/geometry_placeholder.yaml` (**all values placeholders**): linear sweep, ±15°, 41 projections, source height 1000 mm, stationary detector. Detector binned 16× (192², 2.224 mm) for the smoke test and 4× (768², 0.556 mm) for the benchmark.

Legend: **[M]** measured in this session · **[R]** read from source/docs · **[I]** inferred, not measured

## 1. Installed versions

| Component | Version | Notes |
|---|---|---|
| Python | 3.12.13 (uv-managed venv `.venv`) | |
| LEAP (leapct) | 1.26, commit `0c8846f`, CPU-only, **locally patched** | built by `third_party/leap/build_leap_mac.sh` |
| libomp | 23.1.3 (Homebrew) | Homebrew installed this session |
| Compiler | Apple clang 21.0.0 | Xcode toolchain |
| CMake | 4.4.4 (pip, in venv; build only) | |
| numpy / scipy / matplotlib | 2.5.3 / 1.18.1 / 3.11.2 | |
| pyyaml / pydantic / tifffile | 6.0.3 / 2.14.0 / 2026.9.20 | |
| pytest / imageio | 9.1.1 / 2.38.1 | imageio approved as new dependency |

Not installed: napari, pydicom (declared as optional extras), PyTorch/leaptorch, TIGRE.

## 2. LEAP installation — what failed and how it was fixed

| # | Problem | Fix | |
|---|---|---|---|
| 1 | `setup.py` lists PyTorch as a dependency | Install only the ctypes modules (`leapctype.py`, `leap_filter_sequence.py`, `leap_preprocessing_algorithms.py`) + `libleapct.dylib`, as LEAP's `manual_install.py` does | [R] |
| 2 | Default `src/CMakeLists.txt` requires CUDA | Use LEAP's own `src/cpu_CMakeLists.txt` | [R] |
| 3 | CMake 4 rejects `cmake_minimum_required(VERSION 3.0)` | `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` | [M] |
| 4 | `Could NOT find OpenMP_C` (Apple clang has no OpenMP) | Homebrew `libomp` + explicit `OpenMP_*` flags | [M] |
| 5 | Compile errors: `zeroPadForOffsetScan_GPU` and `cudaFree` undeclared in `filtered_backprojection.cpp` (lines 282, 459, 505) | 3 `#ifndef __USE_CPU` guards (`third_party/leap/cpu_build_fix.patch`). Introduced by upstream commit `f884b4c` (2024-11-03); present in v1.24–v1.26 and `main` | [M] |
| 6 | `ModuleNotFoundError: imageio` (top-level import in `leapctype.py`) | Added imageio | [M] |

After the fixes the build is clean apart from 13 compiler warnings (not reviewed). `lp.version()` = 1.26; `get_gpu()` = −1 (CPU) [M].
LEAP license: MIT (LICENSE file, © LLNS) [R]. Whether it fits ViBo's commercial plans is not assessed here. TIGRE's license is still unchecked.

## 3. LEAP behaviors found with our geometry (important for M1/M3)

1. **Cubic voxels are forced.** For non-axially-aligned modular-beam geometry, LEAP silently sets voxel height = voxel width and only prints a warning [M].
   The first smoke run (requested dz = 2.0 mm, xy = 1.84 mm) was therefore wrong. Because the same wrong grid was used to project and to reconstruct, its bead depths looked perfect. That run is discarded. The adapter now reads back LEAP's volume parameters and raises on any mismatch, and a test guards this.
   *Consequence:* the config's 0.46 × 0.46 × 1.0 mm grid cannot be used directly. This needs a decision before M3.
2. **−0.5 voxel x/y offset.** LEAP's modular-beam projector places voxels half a voxel off in x and y compared with its documented `set_volume` formula. Evidence [M]:
   - The offset is −0.50 ± 0.01 voxel for 2 mm and 3 mm voxels.
   - It's the same for even and odd volume and detector sizes.
   - In z it is < 0.1 mm.

   The root cause in LEAP's source was not identified. The adapter compensates. Afterwards, single-voxel and bead centroids agree with the analytic ray–detector intersection to ≤ 0.034 px. With the correction disabled, `test_projection_geometry` fails at 0.66 px [M].
   This is not yet the M1 round-trip test, which needs the NumPy reference projector, not built yet.
3. **LEAP FBP is unsupported for this geometry:** "Error: FBP only implemented for modular geometries whose rowVectors are aligned with the z-axis" [M].
   As a stopgap, M0 uses a NumPy ramp × Hann filter along the sweep direction plus LEAP's backprojector (`recon/simple_fbp.py`). It has no slice-thickness filter, cosine weighting or amplitude calibration, so it is good for localization only.

## 4. Smoke test (64³, 41 projections) — `scripts/m0_smoke.py`

Volume: 64³ cubic 2.0 mm voxels, z = 20–146 mm above the detector. Phantom: a sphere (r = 35 mm, μ = 0.02 /mm) plus 6 beads (r = 2.5 mm, μ = 0.5 /mm). Reconstructions: ramp+LEAP BP, and OS-SART (LEAP `SART`, 10 iterations, 8 subsets).

Bead depth (peak of the z-profile through the true bead x,y, with parabolic sub-slice refinement); slice = 2 mm [M]:

| Bead (x, y, z) mm | ramp+LEAP BP error | OS-SART error |
|---|---|---|
| (−40, 0, 40) | −0.23 slice | +0.06 |
| (−20, 0, 66) | +0.01 | +0.17 |
| (0, 0, 84) | +0.03 | −0.04 |
| (20, 0, 104) | +0.09 | +0.13 |
| (40, 0, 128) | +0.22 | +0.04 |
| (0, 30, 56) | +0.32 | +0.13 |

**Result: all beads appear at the correct depth, within ±0.32 slices (±1 slice criterion) for both methods [M].**
The test is not fully independent: the phantom, projector and reconstructor share the voxel grid, and only the analytic projection check above is independent of LEAP.
Timing: projection 0.013 s, ramp+BP 0.10 s, OS-SART 10 it 1.9 s [M]. Projection-check residual ≤ 0.034 px (columns) and ≤ 0.022 px (rows) [M].

Figures:
- `figures/m0_smoke_xz.png`: xz slice at y = 0, with true bead centers marked
- `figures/m0_smoke_xy.png`: in-plane slices at three bead depths
- `figures/m0_smoke_zprofiles.png`: z-profiles through each bead
- `figures/m0_smoke_projection.png`, `figures/m0_geometry.png`

Visual check:
- Beads show the expected limited-angle elongation along z and X-shaped streaks.
- The ramp+BP z-profiles have negative lobes, as expected without a slice-thickness filter.

## 5. Benchmark (256³, 41 projections, detector 768²) — `scripts/m0_benchmark.py`

The volume is 256³ cubic 0.46 mm voxels (118 mm cube). Each case runs in a fresh process. Time per iteration = (t₃ − t₁)/2, which removes the setup cost. Three runs [M]:

| Case | Run 0 | Run 1 | Run 2 | Peak RSS | CPU/wall |
|---|---|---|---|---|---|
| Forward projection | 0.73 s | 1.04 s | 1.44 s | 0.55 GB | 11–13 |
| Backprojection | — | 13.6 s | 11.7 s | 0.46 GB | 10–12 |
| **FBP-equivalent** (ramp+LEAP BP) | 6.5 s | 14.3 s | 12.3 s | 2.04 GB | 9.5–14.5 |
| LEAP FBP | fails (unsupported) | | | | |
| **OS-SART, 8 subsets, per iteration** | 19.2 s | 26.3 s | 23.8 s | 0.93 GB | 10–13.5 |
| SIRT, per iteration | 14.7 s | 11.7 s | 11.3 s | 0.83 GB | 12–14 |

- **Threads:** LEAP calls `omp_set_num_threads(omp_get_num_procs())`, which is 18 here [R]. It ignores `OMP_NUM_THREADS` [I, from source]. The measured effective parallelism (CPU time / wall time) is 10–14.5 [M].
- **Peak memory:** RSS ≤ 2.04 GB at 256³ [M]. The ramp-filter peak comes from my float64 FFT padding in NumPy, not from LEAP [I].
- **Run-to-run spread is large:** up to 2× for FBP-equivalent, 1.4× for OS-SART. The machine was not idle: a browser process used about 1 core, and the load average was about 24 during the runs [M]. Treat these as typical desktop numbers, not best case.

**Observations:**
- Backprojection is 8–16× slower than forward projection at this size [M]. The cause is not investigated; it might be the CPU modular-beam backprojector or the thread scheduling [I].
- OS-SART is about 2× slower per iteration than SIRT, although each iteration does the same projection work [M]. The likely cause is per-subset overhead in LEAP's Python SART loop [I, not profiled]. Per iteration, OS-SART should still converge faster than SIRT [I].
- **For the brief's M0 question, the measured 256³ OS-SART cost is about 19–26 s per iteration on this Mac.** A 10-iteration run takes roughly 3–4.5 min [I, extrapolated from the per-iteration time].

  How this scales to full-size volumes (more voxels, finer detector) was not measured. A plain voxel-count scaling would put a 768² × 300 volume (10.5× the voxels) at roughly 10× the 256³ cost per iteration, about 3–4 min per iteration, but that is [I] only. Measure it before deciding how much of M3–M5 stays local.

## 6. Tests and repo

- `pytest`: 7 tests, all pass in 0.9 s [M].
  - Geometry: config loading, validation failures, analytic projection and parallax, volume axes.
  - LEAP: rejects non-cubic voxels; projection geometry < 0.1 px; OS-SART bead depths within ±1 slice at 48³.
  - The LEAP tests skip themselves if leapct is missing.
- I checked that the projection test catches the half-voxel offset when the correction is removed [M].
- Git repository initialized with a first commit.

## 7. Open items

1. ~~Non-cubic voxels~~ — resolved: reconstruct cubic, resample in z (docs/decisions.md item 7). Resampling step not yet implemented.
2. Find the root cause of LEAP's −0.5 voxel offset. Possibly report it upstream together with the CPU build bug (needs Gil's OK; not filed).
3. Build the NumPy reference projector and the full round-trip test (M1). The current check uses single analytic points.
4. Re-run the benchmark on an idle machine. Profile backprojection and OS-SART subset overhead. Measure larger volumes.
5. GPU part of M0 (TIGRE + LEAP on a rented NVIDIA GPU) and the TIGRE license check.
6. CI is not set up: there's no remote or CI service yet.
