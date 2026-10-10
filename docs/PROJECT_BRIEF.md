# Digital Tomosynthesis Reconstruction Tool — Project Brief

**Owner:** Gil Travish (ViBo Health)
**Version:** 0.3 (2026-10-09) — device is the NextGen PANEL moving-source prototype: 2D source scan (not a linear sweep); first use case veterinary dental. M0 (Mac) results added.
**Status:** Pass A (simulation), M0 Mac side done; GPU side of M0 not started

**Hardware references** (commercial in confidence; not copied into this repo):
- *Panel X-ray Prototype Requirements: Moving Source Digital Tomosynthesis System*, v2, July 2025 — `~/Documents/Projects/Panel/Hardware_Panel/Panel Prototype Hardware Requirements v2.pdf`
- *Suggested Measurements for the Moving-Source Digital Tomosynthesis Prototype*, 15 July 2026 — `~/Documents/Projects/Panel/Hardware_Panel/Suggested Measurements for the Panel-ULIV Prototype.pdf`

---

## 1. Purpose

Build a reconstruction tool for digital tomosynthesis (DTS) images from the NextGen PANEL moving-source DTS prototype (being built with the University of Liverpool). The x-ray source is moved over a **2D grid of positions in a plane above a stationary detector**, which differs from linear-sweep chest DTS and from mammography. The **first use case is veterinary dental imaging**; the hardware requirements also list NDE (composites, metal AM parts, electronics) and large-animal chest. The prototype is also meant to generate well-documented projection datasets for later (including learned) reconstruction work.

The tool must be ready when bunker (x-ray) access becomes available. The hardware is being developed in parallel, so the software is built and validated on simulated data first.

## 2. Development strategy

**Pass A — simulation (now).** Run the full pipeline on simulated phantoms with placeholder geometry: geometry model, simulator, reconstruction, metrics, calibration, I/O and tool. Pass A produces:

1. A validated reconstruction package.
2. Trade-study results that feed back into hardware design (angular range, number of projections, motion mode).
3. Requirements for the hardware team: geometry accuracy tolerance, required per-frame metadata.
4. A bunker test plan with predicted results for each acquisition.

**Pass B — real data (bunker).** Swap in as-built geometry and detector parameters, re-run the simulation predictions, acquire, calibrate, and compare against predictions.

Rule: every component must run on simulated data with a known ground truth before it touches real data.

## 3. Technical decisions

| Decision | Choice | Status / rationale |
|---|---|---|
| Language | Python ≥ 3.10 | Decided |
| Primary reconstruction engine | **LEAP** (LLNL) v1.26 | Runs on the Mac in CPU mode and on NVIDIA GPUs, so one engine works everywhere. MIT license (confirmed from its LICENSE file); "modular-beam" per-projection geometry handles the 2D source grid directly; SIRT, SART, RWLS, MLEM/OSEM, ASD-POCS, TV filters; PyTorch bindings for possible learned methods later. The Mac CPU build carries local patches (build fix, projector half-voxel bug fix, backprojection speedup; `docs/decisions.md` items 2, 6, 9, 12). Limits for our geometry: cubic voxels only (reconstruct cubic, resample in z); LEAP's built-in FBP does not support it (we filter ourselves and use its backprojector) |
| Secondary engine | **TIGRE** (v3+) | CUDA only, so cloud GPU only. Kept for its larger algorithm set (FISTA, AwTV, SART-TV, Krylov methods) and documented tomosynthesis use, and as a cross-check on LEAP |
| Engine coupling | Thin adapter layer; nothing outside `engines/` imports TIGRE or LEAP directly | Allows engine swaps and cross-checks |
| CPU reference projector | Small, slow, pure NumPy ray-driven projector | Ground truth for geometry tests and CPU-only CI |
| GPU | NVIDIA + CUDA for TIGRE and full-resolution work, rented by the hour | Apple GPUs (Metal) are not supported by either library |
| Excluded | Shift-and-add as a deliverable method | Too limited; may be kept as a debugging aid only |

### 3.1 Compute setup

| Machine | Role | Limits |
|---|---|---|
| Gil's MacBook Pro, Apple M5 Max, 36 GB unified memory | Day-to-day development with Claude Code; CPU tests; LEAP (CPU mode) | No CUDA. Measured in M0 (linear placeholder geometry, 41 projections, 768² detector, patched LEAP): FBP-type reconstruction (filter + backprojection) of 768 × 768 × 652 voxels in 29 s, 3.3 GB; OS-SART 2.3 s/iteration at 256³. Iterative methods that keep several whole-volume copies are the memory limit (one copy per OS-SART subset when sensitivities are cached) |
| Rented NVIDIA GPU, 24 GB class (RTX 4090 / L4), on demand | TIGRE, `pytest -m gpu`, full-resolution runs | Paid by the hour; stop when idle |
| Rented A100 80 GB, on demand | Large M4/M5 trade-study sweeps | Paid by the hour |

M0 result: FBP-type reconstruction is fast enough on the Mac at study sizes (well under the ~2 min a 2010-era CPU system took), so no GPU is needed for it. Iterative cost scales with volume size × projections (about 1 min per OS-SART iteration at a 768 × 768 × 652 volume, extrapolated, not measured). Decide per study in M4 whether a sweep stays local.

**Open item:** TIGRE's license is not verified. It is believed to be BSD-3. LEAP is MIT (LICENSE file read). Check both against the project's commercial plans before building on them (M0).

## 4. Geometry convention and config (proposed; confirm with hardware team)

### 4.1 World frame

- Units: **mm** for lengths, **degrees** for angles in human-facing config. Internal code may use radians.
- Origin: center of the detector front face (the detector is stationary).
- **+z**: normal to the nominal detector plane, pointing toward the source. Reconstructed slices are planes of constant z.
- **+x**: along the detector's column index direction (its long axis; for the dental detector, the 30 cm side). For linear sweeps this is also the sweep direction. (Clarified in v0.3 for 2D source scans; previously "nominal sweep direction". See `docs/decisions.md` item 15.)
- **+y**: completes a right-handed frame.

### 4.2 Per-projection description (engine-independent)

Each projection *i* has:

| Field | Shape | Meaning |
|---|---|---|
| `src` | (3,) | Focal spot position |
| `det_center` | (3,) | Center of the detector front face |
| `det_u` | (3,) unit | Direction of increasing column index |
| `det_v` | (3,) unit | Direction of increasing row index |
| `timestamp` | scalar | Acquisition time (s), optional |
| `exposure` | dict | kVp, mAs, filtration, optional |

Detector-wide fields: `n_cols`, `n_rows`, `pixel_pitch_u`, `pixel_pitch_v`.

Two ways to supply geometry:

1. **Parametric:** 2D source **grid** (primary: grid shape, extent, source height, acquisition order), or linear / arc sweep (for comparison: angular range, number of projections, distances). Expanded to per-projection vectors on load.
2. **Explicit:** per-projection vectors from calibration or encoder logs. This is the Pass B path.

Internally, geometry is always the explicit form. See `configs/geometry_dental.yaml` (device, approximate) and `configs/geometry_placeholder.yaml` (linear sweep, for comparison).

### 4.3 Mandatory geometry checks

- Validate on load: `det_u ⊥ det_v`, both unit length, source on the +z side of the detector, no NaNs.
- 3D plot of the trajectory and detector poses (`dts plot-geometry`).
- **Round-trip test:** project a bead phantom with the NumPy reference projector and with each engine adapter. Bead centroids must agree to < 0.1 pixel. Converting general vectors into TIGRE's parameters (per-angle DSO/DSD, offsets, detector rotation, Euler angles) is the highest-risk conversion in the project; this test guards it.

## 5. Simulation scope

**Phantoms**

1. **Geometric:** bead grid at known (x, y, z), thin wire, slab stack at known depths, edge or disc for in-plane MTF, low-contrast discs.
2. **Anatomical (procedural):** first a **dental phantom**: jaw-like bone with teeth (enamel, dentin, pulp canal, periodontal space, alveolar bone) in soft tissue, parameterized by size from cat to large dog. Later, for the other listed use cases: a canine-thorax-like phantom (lung, soft tissue, fat, ribs, spine) and NDE-like objects (layered composite with voids, PCB-like fine detail).
3. **Calibration phantom:** bead pattern designed in M5.

**Forward-model effects, added in stages**

| Stage | Effects |
|---|---|
| S0 | Monoenergetic line integrals, no noise |
| S1 | Poisson quantum noise at a set I₀; additive electronic noise |
| S2 | Polyenergetic spectrum → beam hardening (two-material) |
| S3 | Scatter (convolution-kernel model, scatter-to-primary ratio by thickness) |
| S4 | Focal-spot blur; source motion blur during exposure (continuous-motion mode) |
| S5 | Truncation (object wider than the detector's field of view; with a 2D source grid wider than the detector, the upper part of a tall object is not seen by every source position); detector lag (placeholder) |

Pass A results must state which stages were enabled. Simulation fidelity is limited, so image-quality numbers from simulation are **upper bounds**.

## 6. Reconstruction methods

1. **DTS-FBP:** ramp filter × slice-thickness filter × apodization (Hann or similar), then the engine's backprojector. Filter parameters are configurable. Library FDK is not adequate for limited-angle data without this filter design. With a 2D source grid there is no single sweep direction: the filter must be 2D (isotropic, or oriented per projection along that source's offset). An isotropic 2D ramp is the M0 stopgap.
2. **OS-SART:** baseline iterative method.
3. **TV-regularized:** ASD-POCS first, then optionally FISTA-TV. Watch for blocky "cartoon" texture on soft tissue.
4. **Truncation handling:** extended reconstruction volume and/or projection extrapolation; masking.

## 7. Metrics

All metrics are computed against ground truth in simulation.

| Metric | Measures |
|---|---|
| Artifact spread function (ASF) | Out-of-plane blur of a bead along z; with a sparse 2D source grid, also the discrete "ghost lattice" of out-of-plane copies |
| Slice sensitivity profile (SSP) FWHM | Effective slice thickness |
| In-plane MTF (edge method) | In-plane resolution |
| CNR on low-contrast discs | Detectability, as a proxy |
| RMSE / SSIM in in-focus planes | Global fidelity, used with caution under limited angle |
| Runtime, peak GPU memory | Practicality |

Clinical image review is required before choosing a method. Metrics alone are not sufficient.

## 8. Milestones

Each milestone ends with a written result: a short report in `reports/` with figures and numbers.

| ID | Milestone | Acceptance criterion |
|---|---|---|
| M0 | Environment and repo: LEAP (CPU) on the Mac; TIGRE + LEAP on a rented NVIDIA GPU; license check; CI running CPU tests; Mac speed benchmark | `pytest` green on the Mac; 64³ smoke test passes with LEAP on the Mac and with both engines on the GPU; 256³ OS-SART runtime on the Mac recorded |
| M1 | Geometry: schema, loader, validator, plotter, parametric generators (grid, linear, arc), explicit (encoder-log) geometry, TIGRE and LEAP adapters | Round-trip test < 0.1 px for both engines on grid and linear trajectories |
| M2 | Simulator: phantoms + stages S0–S2 | Simulated projections match reference projector (S0) to numerical precision; noise statistics match theory (S1) |
| M3 | Reconstruction: DTS-FBP, OS-SART, ASD-POCS | Beads reconstruct at correct depths (±1 slice) in all methods |
| M4 | Metrics + comparison report; add S3–S5; trade-study harness | Report comparing methods across ≥ 2 phantoms and ≥ 3 geometries (grid density, e.g. 5 × 5 vs 6 × 6, and ~10 / 20 / 40 projections; source extent; SID; motion mode) |
| M5 | Simulated calibration: phantom design, geometry estimator, sensitivity study | Estimator recovers injected errors within spec; report states the geometry tolerance needed to keep metric loss < X % (X set by Gil) |
| M6 | I/O and tool: mock raw format, dark/flat/log preprocessing, CLI, TIFF/DICOM export, napari viewer | End-to-end run from mock raw files to slice stack by someone other than the developer |
| M7 | Bunker test plan, aligned with the *Suggested Measurements* document (commissioning, basic imaging, representative objects, datasets at ~10/20/40 projections and several exposures, full metadata) | Ordered acquisition list with predicted results and pass/fail thresholds |
| M8 (optional) | Learned post-processing or unrolled reconstruction via LEAP + PyTorch | Only if M4 shows depth blur or noise is the limiting factor |

## 9. Repository layout (proposed)

```
dts-recon/
  CLAUDE.md
  README.md
  pyproject.toml
  configs/              geometry and run configs (YAML)
  docs/                 this brief, interface spec, decisions log
  src/dtsrecon/
    geometry/           schema, loader, validation, generators, plotting
    engines/            tigre_adapter.py, leap_adapter.py, reference_projector.py
    phantoms/           geometric, anatomical, calibration
    simulate/           forward model stages S0–S5
    recon/              dts_fbp.py, iterative.py, truncation.py
    metrics/            asf, ssp, mtf, cnr, fidelity
    calibration/        geometry estimation
    io/                 raw readers, preprocessing, export
    cli.py
  tests/                CPU tests by default; GPU tests marked @pytest.mark.gpu
  scripts/              studies and report generation
  reports/              milestone reports (figures + markdown)
```

## 10. Interface with hardware team (to be agreed)

Items the hardware team must provide, or agree to, early:

1. Raw detector output format: file type, bit depth, header fields, byte order.
2. Dark and flat-field acquisition procedure.
3. Per-frame geometry metadata: encoder (x, y) source position and timestamp per projection, exposure (kV, current, time), number of exposures per location. Also whether the source moves during exposure. Output format per the requirements: DICOM or raw projection sets with metadata.
4. Trajectory type and nominal parameters (section 11).
5. A single owner for the interface spec (`docs/interface_spec.md`). Changes go through that document.

## 11. Placeholder parameters (all to be replaced)

Approximate values from Gil (2026-10-09) unless marked; the hardware requirements document is cited where it sets a range. **None has been measured on the device.**

| Parameter | Current value | Source |
|---|---|---|
| Use case | Veterinary dental (first); NDE and large-animal chest later | Gil; requirements doc |
| Trajectory | 2D source grid in a plane parallel to the detector; stationary detector ("never moves") | Gil |
| Grid | 5 × 5 or 6 × 6 positions over ≥ 300 × 300 mm, maybe 400 × 400 mm (larger than the detector). Config default: 6 × 6 over 400 × 400 mm, serpentine order | Gil (approximate); order is a placeholder |
| Number of projections | 25–36 (study ~10 / 20 / 40) | Gil; measurements doc; requirements: ≥ 20 |
| Source-to-detector distance | 500 mm | Gil (approximate) |
| Angular coverage (from detector center) | ±21.8° per axis, 29.5° at corners (6 × 6 over 400 mm at SID 500) | computed |
| Object-to-detector gap | 10 mm | Gil (approximate) |
| Object height | up to 200 mm | Gil (approximate) |
| Detector | 300 × 200 mm, 0.1 mm pixels (3000 × 2000); study binning 4 → 0.4 mm | Gil (approximate); binning is a placeholder |
| Motion mode | Step-and-shoot (also study continuous) | placeholder |
| Source (selected) | NSI / Rigaku **M237** monoblock: 35–70 kV (custom 15–80), 10 W max, 5–200 µA, **transmission** target (Au, Ag, Rh or W — which one TBD), Be window (thickness not stated), 110° cone, tungsten-filament cathode, grounded anode, air cooled, < 700 g. Focal spot size and spectrum not stated | rigaku.com product page, read 2026-10-09 |
| Source (requirements) | 60–120 kV, 10 W, field emission, spot < 1 mm (2–3 mm minimum spec), ~90° cone | requirements doc |
| Detector type | CMOS flat panel, 14–16 bit, ≥ 10 fps, full scan < 30 s | requirements doc |
| kVp for dental | TBD | — |
| Clinical tasks | Dental: tooth roots, periodontal bone, fractures (to be confirmed) | placeholder |

## 12. Risks

| Risk | Mitigation |
|---|---|
| Geometry/calibration error dominates image quality (inference) | M5 sensitivity study sets the tolerance early |
| Errors when converting vectors to TIGRE parameters | Round-trip test against reference projector (M1); LEAP's modular-beam input is a direct match to our format, so it is tested first on the Mac |
| Engine bugs in our geometry (found in M0: LEAP CPU projector was half a voxel off for sources above the detector) | Projector, backprojector and adjoint tests against analytic geometry, kept in `pytest`; same tests on the GPU build |
| Sparse 2D source grid produces discrete out-of-plane "ghost lattices" (seen in M0 simulation) | Study grid density and filter design in M4; include in ASF metric |
| Tall objects only partly sampled when the source grid is wider than the detector | Report the fully sampled region per geometry; truncation handling (§6.4) |
| Mac CPU too slow for iterative work at useful sizes | M0: FBP is fast; iterative ≈ 1 min/iteration at large study volumes (extrapolated). Move large sweeps to a rented GPU if needed |
| Simulation under-models scatter, lag, heel effect | Treat Pass A numbers as upper bounds; report stages used |
| Over-tuning to one phantom | ≥ 2 phantom types; held-out case |
| Interface drift with hardware | Single-owner interface spec |
| License incompatibility with commercial use | Verify in M0 |
| GPU memory limits at full detector size | Detector binning for studies; measure peak memory in M4 |
