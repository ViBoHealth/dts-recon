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
16. **M1 geometry I/O (2026-10-09).** Explicit geometry in YAML (`projections:`) or CSV (`projections_file:`):
    columns `src_x, src_y, src_z` required; detector pose columns optional (default: stationary nominal
    detector); `timestamp`, `kvp`, `ma`, `ms` optional. `save_explicit` archives any geometry exactly. This
    CSV layout is **our proposal only** — the real encoder-log format belongs in the hardware interface spec
    (brief §10), which has a single owner. Example: `configs/examples/`.
17. **Round-trip test (brief §4.3) implemented against an exact analytic reference** (sphere line integrals),
    plus an independent ray-marching voxel projector. LEAP vs reference: max 0.068 px (dental grid) and
    0.036 px (linear) with beads ~9+ px across. Small sharp beads (~5 px) show up to ~0.11 px from pixel-sampling
    bias of the centroid estimator (shown to shrink with bead size and not with voxel size).
18. **M2 simulator design (2026-10-09).** Phantoms are ordered lists of analytic shapes (ellipsoid, box, finite
    cylinder; later shape wins) with materials. The forward model works in per-material path lengths
    (exact analytic route for reference; voxel-fraction route projected by the engine for speed), so S2
    (polyenergetic) only changes the last step. S0/S1 use single-energy PLACEHOLDER attenuation values
    (`simulate/materials.py`, ~60 keV, not tabulated data) and PLACEHOLDER noise (I0 1e4 counts/pixel,
    electronic noise 5 counts rms). Acceptance measured: per-material route vs projecting the mu-volume agrees
    to 5e-7–1e-6 relative (float precision); voxel vs exact route converges with voxel size (1.6 % -> 0.7 %
    at 1.0 -> 0.5 mm); S1 mean and variance match small-noise theory within 0–0.8 % at >= ~400 counts and
    deviate (4–260 %) at low counts, as expected (approximation + clipping before the log).
19. **Dental phantom** (`phantoms/dental.py`): lateral head on the detector, near/far jaw halves at different
    depths, 5 teeth per jaw quarter (enamel, dentin, pulp, ligament space, roots in cortical/trabecular bone),
    cranium shell, nasal air; scales cat / small dog / medium dog / large dog. Anatomy is PLACEHOLDER. By
    default asymmetric: a periapical lesion (near mandible, tooth 2) and a missing tooth with empty socket
    (far mandible, tooth 3), recorded as ground truth — targets for M3/M4.
20. **S2 polyenergetic model (2026-10-09).** Source: M237, W transmission target, 70 kV, no added filtration
    (Gil; 1–2 mm Al optional), focal spot ~0.5 mm (Gil, from past units), Be window negligible. Attenuation:
    xraydb (new dependency, approved) with NIST/ICRU-44 compositions for soft tissue, cortical bone, adipose,
    water, PMMA, CsI (read from NIST Table 2); reproduces NIST cortical-bone mu/rho to 0.01 % (20–80 keV).
    Enamel, dentin, trabecular bone are PLACEHOLDER mixtures. Spectrum: Kramers bremsstrahlung, 5 um W
    self-filtration (PLACEHOLDER), energy-integrating 0.5 mm CsI (PLACEHOLDER); mean detected energy 33.9 keV
    unfiltered, 38.9 / 41.0 keV with 1 / 2 mm Al. S0/S1 single-energy values are now derived from this
    spectrum after 40 mm soft tissue (local slope; mean 42.6 keV), replacing the earlier placeholder table.
    Dental simulation (medium dog, OS-SART 8 it): beam hardening lowers reconstructed hard-tissue values by
    ~26–36 % vs S0 (enamel 0.107 -> 0.069 /mm); all DTS values are far below true mu for thin dense structures
    (limited angle + partial volume), so reconstructed values are not quantitative. Max S0 line integral 9.1
    (transmission ~1e-4: photon starvation with the placeholder I0) vs 6.1 for S2.
21. **M3 reconstruction (2026-10-09).**
    - LEAP's TV (`diffuse`, `TVcost`) is GPU-only in the CPU build ("GPU routines not included"), so LEAP's
      ASDPOCS on the Mac silently degenerates to SART. We implement ASD-POCS (Sidky & Pan 2008) with a NumPy
      TV (gradient checked by finite differences) and our cached OS-SART operator as the data step.
    - DTS-FBP (`recon/dts_fbp.py`): cosine weight about each source's foot point; ramp (radial or oriented along
      the source offset) with a low-frequency floor (dc), Hann apodization, optional slice-thickness filter;
      LEAP backprojection / N. Output not calibrated to mu.
    - Acceptance (brief §8, beads at correct depth ±1 slice): 60 steel beads in PMMA, dental 6x6 geometry,
      1 mm voxels, S0 and S1: max error 0.25–0.40 slice for DTS-FBP (radial, oriented), OS-SART and ASD-POCS.
      Out-of-plane spread between beads: ASD-POCS < OS-SART < FBP radial < FBP oriented (visual + z profiles).
    - Dental (S2+S1, medium dog, 0.8 mm): the missing tooth is found at the far-jaw depth (|err| <= 0.9 mm) and
      separated from the near-side tooth at the same (x, y); the small periapical lesion is found 1.5–2.3 mm off
      (within its ~2.6 mm radius, but 2–3 slices) — a detectability/depth-resolution question for M4.
    - Observed: DTS-FBP gain rises with height (identical beads 0.7 at z=35 vs 1.0 at z=125 mm): a depth-dependent
      weighting to characterize and correct in M4.
    - Depth metric windows are now in mm (a voxel-count window reached the next bead at coarse voxels).
22. **M4 (2026-10-09).** Metrics (`metrics/iq.py`), stages S3–S5 (`simulate/acquisition.py`), trade study
    (`scripts/m4_study.py`, results in `reports/m4/`): 8 grid geometries x DTS-FBP / OS-SART / ASD-POCS x
    IQ + dental phantoms, realistic stages; baseline also S0+S1 and +1.5 mm Al. All physics PLACEHOLDER;
    noise not dose-matched (fixed I0 per projection).
    - Disc-edge MTF dropped: DTS loses the low-frequency interior of wide thin objects (edge overshoot
      150–200 %), so the f=0-normalized edge MTF is meaningless. In-plane resolution = bead FWHM instead.
    - ASD-POCS defaults were mistuned for realistic data (depth error 2.2 slices, ghost ratio up to 2.8);
      tuned on the IQ phantom at 6x6/400 (alpha 0.05, 10 TV steps, w_z 0.25) — home advantage at that case.
    - Results (realistic): all methods locate IQ beads within 0.75 slice (ASD-POCS 3x3 worst). Depth
      resolution (ASF FWHM) ASD-POCS 2.3–5.0 mm < OS-SART 5.2–6.4 < DTS-FBP 5.2–7.9; 400 mm source extent beats
      300 mm and SID 600 (larger angle). Ghost ratio falls with projection count (FBP 0.33 at 9 -> 0.15 at 36)
      and is lowest for ASD-POCS (~0.08 at >= 20 projections). Diminishing returns beyond ~25 projections.
      In-plane bead FWHM ~2.0–2.2 mm everywhere (detector/voxel limited at 0.8 mm binning).
    - Dental: DTS-FBP (radial, no slice-thickness filter) finds the strongest lucency at the wrong jaw for the
      periapical lesion in 5 of 8 realistic geometries; OS-SART and ASD-POCS find it (within 2.5 mm) in all.
      The socket is mislocalized/low-CNR at 300 mm extent (smaller angle -> poor depth separation of jaws).
      1.5 mm Al changed IQ metrics little (not dose-matched).
23. **M4 gaps (2026-10-10)** — `scripts/m4_gaps.py`, results `reports/m4/gaps_{motion,st,trunc}.csv`; 6x6 / 400 /
    SID 500, realistic stages, PLACEHOLDER physics.
    - **Motion mode: step-and-shoot only** (Gil: shooting while moving is not contemplated, too many
      variables). Brief updated. The continuous-motion model (S4) stays in the simulator, tested. For the
      record (IQ phantom): exposure over 25 % of the step costs little; 50 % doubles ghosts (OS-SART 0.16 ->
      0.42) and widens beads 2.0 -> 2.7 mm; 100 % breaks ASD-POCS (bead 5.7 slices off, ghost ratio 4).
    - **DTS-FBP slice-thickness filter** (st 1 -> 0.25, radial and oriented): thickens slices (ASF 6.1 -> 12.4
      mm radial) and raises ghosts; no metric improves meaningfully. Kept off by default.
    - **Correction to item 22 (dental "wrong jaw").** The depth metric "strongest lucency in the head at the
      lesion's (x, y)" is ill-posed: the ground truth itself has comparable lucencies in the other jaw at the
      same (x, y) (pulp / ligament space). Truth "prominence" (contrast at the true depth / strongest lucency
      elsewhere in the head) is only 1.05 for the periapical lesion and 0.54 for the socket (the truth's
      strongest lucency at the socket's (x, y) is in the near jaw). So DTS-FBP did not put the lesion in the
      wrong jaw; it shows it with slightly less contrast relative to that anatomy (prominence 0.88 vs
      OS-SART 1.41, ASD-POCS 1.96). Added `*_local_depth_err_mm` (peak within ±8 mm of the truth) and
      `*_prominence`. At 6x6/400 all methods place both targets within 2.3 mm (local); at 5x5/300 the socket is
      6–8 mm off for DTS-FBP and OS-SART (real depth-resolution loss at the smaller angle; ASD-POCS 1.5 mm).
      Dental lesion conspicuity needs image review, not this metric alone.
    - **Truncation** (tall wide PMMA block, z 15–195 mm, 320 x 240 mm: wider than the detector, taller than the
      ~167 mm fully sampled height; targets at 40 / 100 / 150 / 185 mm):
      * Iterative methods need the extended volume (whole object). Volume cut at the fully sampled height and
        detector width: OS-SART / ASD-POCS ghost ratio 1–4.8, bead depth errors up to 3.1 mm, background
        spread 93 %. Extended volume: ghost 0.14–0.36, depth error <= 0.6 mm, spread ~10 %; ASD-POCS keeps
        ASF ~4.1–4.7 mm and depth error < 0.45 mm even at z = 185 (seen by 24 of 36 sources).
      * DTS-FBP is independent of the volume extent (per-voxel backprojection), but zero padding at the
        detector edge rings badly (ghost ratio 2–6); edge extrapolation (`edge_taper_mm=40`) cuts it 5–8x
        (0.5–0.8) and the background spread 47 -> 37 %. Still far worse than iterative + extended volume.
      * The 12 mm fat discs are not detected by DTS-FBP (wrong-sign CNR): wide, low-contrast objects lose
        their low frequencies in DTS.
    - **RMSE / SSIM** (`metrics.iq.fidelity`, affine-fitted to truth): implemented, but uninformative on these
      phantoms (rmse_rel 0.8–1.0, SSIM 0.95–0.99 on IQ planes, dominated by uniform PMMA that DTS cannot
      reproduce), as the brief warned. Reported, not used for decisions.
24. **M5 geometry calibration and sensitivity (2026-10-10)** — `scripts/m5_study.py`, `scripts/m5_plots.py`, results
    `reports/m5/{calibration,sensitivity,closed_loop,tolerance}.csv`, `sensitivity.png`. All physics PLACEHOLDER.
    - **Method.** Two-plane steel-bead phantom (`phantoms/calibration.py`; beads r = 1 mm at z = 10 and 120 mm,
      6 + 13 beads, min shadow gap 55 px) scanned offline; centroids measured on 0.1 mm windows; least squares
      for every source position (3N) + phantom pose (6), sparse Jacobian, two passes (wide then tight search
      windows). Detector = world frame, so all errors are source errors. Assumes a repeatable stage (the scan
      calibrates the stage, not each patient exam).
    - **Accuracy** (combined error: 7.7 mm worst source uncalibrated). After calibration, 6x6/400, rms x/y/z:
      0.022/0.035/0.068 mm at I0 1e3; 0.011/0.010/0.032 at 1e4; 0.002/0.002/0.007 at 1e5 (max 3D 0.18 / 0.08 /
      0.02 mm). 5x5/400 and 6x6/300 similar. Predicted standard errors match the actual errors (usable as a QA
      check on real scans). ~70 s on CPU.
    - **The phantom's own metrology is the limit, not noise.** Bead-position errors σ 0.02 / 0.05 / 0.1 mm give
      source rms x 0.14 / 0.36 / 0.71 mm (max 3D 0.3 / 0.6 / 1.2 mm): ~7x amplification (plane separation 110
      mm vs SID 500). Recommendation: CMM-measured phantom (~0.02 mm), rather than also estimating bead positions.
    - **Sensitivity** (one error at a time, IQ phantom, beads at 30–70 mm, 1 mm voxels, nominal-geometry recon).
      Offset / SID / scale mainly *distort* (shift targets): offset 10 mm -> 1.2–1.4 mm; SID 20 mm -> 2.6–2.9 mm
      in depth; scale 2 % -> 1.0–1.2 mm. Rotation and tilt *blur*: rotation 4° -> ghost ratio 1.19 (OS-SART) /
      0.93 (ASD-POCS) vs ~0.05 / 0.012; tilt 4° -> ASF +15–20 %. Random jitter up to 4 mm lateral / 8 mm height:
      < 5 % ASF, < 0.25 mm. Offset 5 mm also raises ASF 8–17 % (sub-voxel shifts at 1 mm voxels).
      Tolerances at blur <= 10 %, distortion <= 0.5 mm (set by Gil 2026-10-10, item 27): offset 2 mm,
      SID 2 mm, scale 0.5 %, rotation 1°, tilt 2°, jitter >= 4 mm / >= 8 mm (end of sweep).
    - **Closed loop** (combined error): reconstructing with the estimated geometry gives the same metrics as the
      true geometry (all within 0.001 mm / 0.1 %); the nominal geometry gives 0.43–0.50 mm position errors and
      OS-SART ghost ratio 0.096 vs 0.053.
    - **Caveats.** Distortion and blur scale ~h/(S−h): at 160 mm the effects are ~2.9x those at 70 mm, so the
      tolerances above are loose for tall objects. Finer voxels will expose smaller errors. Calibration
      accuracy (≤ 0.1 mm) is well inside every tolerance either way, so calibrating removes the issue, given a
      repeatable stage and a measured phantom.
25. **M4 compute (2026-10-10)** — `scripts/m4_memory.py`, `reports/m4/compute.csv`. Dental phantom, 6x6 / 400,
    36 projections, Mac M5 Max CPU (LEAP), each case in a fresh process:
    | binning (voxel) | volume | DTS-FBP | OS-SART (10 it) | ASD-POCS (10 it) | peak RSS (sim -> + recon) |
    |---|---|---|---|---|---|
    | 8x (0.8 mm) | 300x175x128 | 5 s | 7 s | 13 s | 0.66 -> 0.68–1.3 GB |
    | 4x (0.4 mm) | 600x350x256 | 40 s | 71 s | 107 s | 3.0 -> 3.5 / 3.6 / 7.9 GB |
    The simulation (voxelization) sets the peak for DTS-FBP and OS-SART; ASD-POCS adds ~4.8 GB at 0.4 mm
    (~24 volume copies: TV gradient temporaries). Decision: the M4/M5 sweeps stay on the Mac (both full studies ran locally); no
    GPU rental needed so far. Inference: a full-detector-footprint volume at
    0.4 mm (~1 GB/copy) puts ASD-POCS near 25 GB, close to the 36 GB limit; reducing its temporaries is the
    first fix if needed.
26. **M6 I/O and tool (2026-10-10)** — `src/dtsrecon/io/{scan,mock,preprocess,export}.py`, `src/dtsrecon/pipeline.py`,
    `dts mock-scan | info | calibrate | recon | view`; user guide `docs/USAGE.md`; raw format `docs/raw_format.md`
    (our PLACEHOLDER strawman for brief §10). End-to-end: `scripts/m6_end_to_end.sh`, scoring `scripts/m6_check.py`,
    logs `reports/m6/`. Dependencies: pydicom (approved list), napari[all] (napari approved; PyQt6 backend
    approved by Gil 2026-10-10).
    - **Raw format:** headerless uint16 frames + `scan.yaml` (darks, flats, projections with position /
      repeat index, encoder x/y, timestamp, kV/µA/ms). Preprocessing: mean dark, mean flat, defect map (flat
      vs 5x5 local median, hot darks), repeats averaged, charge normalization, binning in the linear domain,
      then -ln. Geometry from a calibration file (binned to the scan's pixel size if needed) or encoders.
    - **Mock detector model** (PLACEHOLDER): offset + column offset, smooth and per-pixel gain, read noise,
      dead/hot pixels; conversion gain set so the open beam is <= 60 % of full scale.
    - **End-to-end, mock data** (Mac CPU): calibration scan (36 x 3000 x 2000, 0.1 mm) -> `dts calibrate`:
      every source within 0.05 mm of the truth (encoders up to 7.7 mm off), 600/600 injected defects found,
      5 s. Dental scan (0.4 mm, S2+S3+S4+S5 + detector) -> `dts recon`, volume 850 x 600 x 265 at 0.4 mm
      (detector + 20 mm margin): OS-SART 131 s / 5.4 GB peak, DTS-FBP 90 s / 3.4 GB, ASD-POCS 172 s / 6.9 GB.
      DICOM export written. Lesion and socket found by all methods (reports/m6/e2e_metrics.csv).
      Calibrated vs encoder geometry (OS-SART): same lesion depth error (2.1 mm); the low lesion (z ≈ 37 mm)
      is tolerant of the stage errors, as M5 predicts (h/(S−h)).
    - **ASD-POCS memory fix:** the TV step ran in float64 on whole volumes (~10 volume-sized temporaries):
      18 GB peak at 850 x 600 x 265. Now float32, blocks of 16 slices with a one-slice halo, buffers reused:
      6.9 GB, 270 -> 172 s. Matches the old gradient to 1e-12 in float64 (tests).
    - **ASD-POCS lesion depth is numerically fragile at 0.4 mm.** The lesion's contrast (~0.005 /mm) is
      comparable to the differences between float32 and float64 runs (0.8 % rms of the volume): lesion depth
      error 0.1 mm (float32) vs 0.9 mm (float64) on the same data; an earlier run with the old code gave 5.7 mm
      (not chased further). At 0.8 mm both give 0.3 mm. LEAP itself is deterministic (bit-identical reruns).
      Do not read single-run ASD-POCS lesion depths to better than a few mm; image review still decides.
    - Not done: a run by someone other than the developer (the M6 acceptance test) — needs Gil or a colleague
      to follow docs/USAGE.md.
27. **M5 limits set (Gil, 2026-10-10):** metric loss X = 10 % (ASF and in-plane FWHM growth) and distortion
    D = 0.5 mm (max bead position error), i.e. the values used in item 24 become final. Rationale: 0.5 mm is
    about one voxel at 0.4–0.8 mm and supports measurements on the images (root length, lesion size, bone
    level); 10 % is a common just-noticeable blur change. Calibration (<= 0.1 mm) is well inside both, so the
    limits matter for a stage specification without calibration and for how often to recalibrate. Revisit
    when stage repeatability is known (hardware team) or if image review says otherwise.
