# Decisions log

Entries are dated. Geometry-convention changes need Gil's approval (CLAUDE.md).

## 2026-10-09 — M0 (Mac)

1. **Python 3.12 via uv** (`.venv`). 3.14 is too new for parts of the stack (napari/Qt); 3.12 has wheels for everything listed.
2. **LEAP v1.26 built from source with a local patch** (`third_party/leap/cpu_build_fix.patch`, applied by
   `third_party/leap/build_leap_mac.sh`). v1.24–v1.26 do not compile CPU-only: three GPU-only calls in
   `filtered_backprojection.cpp` lack `#ifndef __USE_CPU` guards. The patch adds those guards; the guarded
   branches only run when a GPU is selected. Approved by Gil. Candidate upstream bug report (not filed).
3. **Homebrew + libomp** for OpenMP (Apple clang has none). Approved by Gil.
4. **imageio** added as a dependency: hard top-level import of `leapctype.py`. Approved by Gil.
5. **leaptorch / PyTorch not installed.** `setup.py` requires torch; we install LEAP's ctypes modules only.
6. **LEAP CPU projector bug fixed by patch** (`third_party/leap/joseph_modular_cpu_fix.patch`, 2026-10-09).
   `projectLine_Joseph` adds a spurious `+0.5` to the x/y sample offsets in its z-dominant, r.z < 0 branch
   (rays toward −z, i.e. our geometry) — present since the CPU modular Joseph projector was added
   (`0683313`, 2023-11-28). Effect: projector −0.5 voxel in x/y; the backprojector was correct, so the two
   disagreed by half a voxel. The adapter's earlier +0.5 voxel offset "correction" fixed the projector but
   shifted the backprojector (and the ramp+BP stopgap) by −0.5 voxel — removed. The adapter now refuses a
   CPU LEAP build without this patch. Guards: `test_projection_geometry`, `test_backprojection_geometry`,
   `test_projector_backprojector_adjoint`. Candidate upstream bug report (not filed).
7. **Cubic voxels required with LEAP** for our geometry; LEAP otherwise silently overrides the voxel
   height. The adapter raises instead.
   - Scope (read from `parameters.cpp:735–745`, v1.26): LEAP forces cubic voxels for every modular-beam
     geometry that is not "axially aligned", and `modularbeamIsAxiallyAligned()` is true only for
     modular geometries converted internally from circular cone-beam (angles present). So **any geometry
     supplied as per-projection vectors** — our parametric expansion and the Pass B calibration path —
     gets cubic voxels regardless of orientation. A LEAP implementation limit (inferred: fast SF
     projector), not a property of tomosynthesis. TIGRE sets voxel size per axis (not verified here).
   - **Resolution (Gil, 2026-10-09):** reconstruct on a cubic grid with LEAP and resample in z to the
     wanted slice spacing. Cost: more slices computed than kept (≈2.2× at 0.46 mm voxels → 1.0 mm slices).
   - For LEAP/TIGRE cross-checks, run both on the same cubic grid or apply the same z resampling to both,
     so slice sensitivity profiles are comparable.
8. **LEAP's FBP does not support our geometry** ("FBP only implemented for modular geometries whose
   rowVectors are aligned with the z-axis"). M0 used a provisional NumPy ramp×Hann filter + LEAP
   backprojector (`recon/simple_fbp.py`). The DTS-FBP of brief §6 (M3) will be built the same way
   (own filters + engine backprojector), which the brief already planned.
9. **LEAP CPU backprojection speedup** (same patch). The CPU modular backprojector searched a ≥ 5×5 detector
   window per voxel per projection, most of it zero weight. The patch hoists loop invariants and bounds the
   search to the exact nonzero footprint (projected ±1-voxel square), intersected with the original window.
   Output **bit-for-bit identical** to unpatched LEAP on 8 geometries (linear, arc, tilted detector, x-, y-,
   mixed-dominant rays). 256³ / 41 proj: backprojection 6.95 → 1.4–1.7 s; OS-SART (8 subsets) 19–26 →
   5.2–5.4 s/iter; SIRT → 2.5–2.6 s/iter. Chosen over an unmatched custom backprojector (option B) by Gil.
10. **Adjoint check:** after the fix, projector and backprojector are adjoint to ~1e-7 for objects away from
    the volume boundary; for content touching the boundary they differ by a few % (LEAP's partial-voxel
    edge handling). Relevant for truncation work (S5): keep a padded volume.
11. **Bead depth metric** (`metrics/depth.py`): half-max 3D centroid, not a fixed z-column. Off-axis bead blur
    is tilted toward the mid-sweep source; a column missing the bead center by a fraction of a voxel reads a
    wrong depth (up to ~1 slice seen). Earlier M0 "ramp+BP within ±0.32 slices" result was partly two
    half-voxel errors cancelling.
12. **OS-SART overhead removed** (2026-10-09). Profiling LEAP's `SART` (256³, 8 subsets) showed two costs:
    (a) the CPU modular projector parallelized over angles only, so a ~5-angle subset used ~5 of 18 threads —
    fixed in `joseph_modular_cpu_fix.patch` (`collapse(2)` over angle × row; forward projections bit-identical
    on 16 geometry/subset cases); (b) `SART` recomputes each subset's sensitivity (a full CPU backprojection)
    every iteration — `LeapEngine.sart` now runs LEAP's update with per-subset sensitivities cached
    (bit-identical to `lp.SART`, tested; `cache_sensitivity=False` falls back to LEAP's loop). Memory: one extra
    volume per subset (+0.47 GB at 256³ / 8 subsets; ~5.7 GB at 768² × 300 — inferred). 256³ / 41 proj:
    OS-SART 5.2–5.4 → 2.3–2.4 s/iter (= SIRT); LEAP's own loop with the threading fix alone: 3.9–4.2 s/iter.
13. **Device geometry is a 2D source scan, not a linear sweep** (Gil, 2026-10-09). Use case: veterinary
    dental. Approximate values: SID 500 mm, stationary 200 × 300 mm detector at 0.1 mm, object up to 200 mm
    tall, ~10 mm above the detector; source on a 5 × 5 or 6 × 6 grid over ≥ 300 × 300 (maybe 400 × 400) mm.
    New `trajectory.type: grid` (serpentine/raster order) and `configs/geometry_dental.yaml` (default 6 × 6 over
    400 × 400 mm). The linear config is kept for comparison. Computed consequences: points above z ≈ 167 mm
    are not seen by every source (20 cm detector side vs ±200 mm sources); a sparse source grid makes
    out-of-plane objects appear as a lattice of discrete ghosts (seen in `scripts/smoke_grid.py`), not a
    continuous blur — relevant to choosing grid density. Stopgap FBP for grids uses an isotropic 2D ramp.
    Reference documents (commercial in confidence, not copied into the repo):
    `~/Documents/Projects/Panel/Hardware_Panel/Panel Prototype Hardware Requirements v2.pdf`,
    `.../Suggested Measurements for the Panel-ULIV Prototype.pdf`.
14. **Depth metric uses the bead's own connected blob** (scipy.ndimage.label) so neighbouring beads inside
    the window cannot bias the centroid; projection check skips (bead, projection) pairs off the detector edge.
15. **Brief v0.3 and +x clarification** (Gil approved the brief update, 2026-10-09). The device is the NextGen
    PANEL moving-source prototype with a 2D source scan; first use case veterinary dental (brief §1, §11).
    §4.1 "+x = nominal sweep direction" no longer defines anything for a 2D grid, so +x is now "along the
    detector's column index direction (long axis); for linear sweeps also the sweep direction". Origin and +z
    unchanged; existing code and configs already follow this (det_u = +x). Wording confirmed by Gil, 2026-10-09.
