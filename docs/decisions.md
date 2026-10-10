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
