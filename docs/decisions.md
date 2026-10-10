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
6. **Adapter compensates a measured −0.5 voxel x/y offset** in LEAP's modular-beam projector
   (`XY_HALF_VOXEL_SHIFT` in `engines/leap_adapter.py`; guarded by `tests/test_leap_engine.py::test_projection_geometry`).
   Root cause not identified in LEAP's source. The world-frame convention (brief §4.1) is unchanged.
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
