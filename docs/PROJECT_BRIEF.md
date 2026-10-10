# Digital Tomosynthesis Reconstruction Tool — Project Brief

**Owner:** Gil Travish (ViBo Health)
**Version:** 0.2 (2026-10-08) — development machine is an Apple M5 Max; LEAP becomes the primary engine
**Status:** Pass A (simulation) not started

---

## 1. Purpose

Build a reconstruction tool for digital tomosynthesis (DTS) images from a moving-source x-ray device for veterinary use. The device geometry differs from mammography and from existing commercial chest DTS. The targets are complex biological anatomy across a wide range of patient sizes.

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
| Primary reconstruction engine | **LEAP** (LLNL) | Runs on the Mac in CPU mode (its README: Mac supported, no GPU) and on NVIDIA GPUs, so one engine works everywhere. MIT license; "modular-beam" per-projection geometry; SIRT, SART, RWLS, MLEM/OSEM, ASD-POCS, TV filters; PyTorch bindings for possible learned methods later. Changed from TIGRE in v0.2; revert if Gil prefers |
| Secondary engine | **TIGRE** (v3+) | CUDA only, so cloud GPU only. Kept for its larger algorithm set (FISTA, AwTV, SART-TV, Krylov methods) and documented tomosynthesis use, and as a cross-check on LEAP |
| Engine coupling | Thin adapter layer; nothing outside `engines/` imports TIGRE or LEAP directly | Allows engine swaps and cross-checks |
| CPU reference projector | Small, slow, pure NumPy ray-driven projector | Ground truth for geometry tests and CPU-only CI |
| GPU | NVIDIA + CUDA for TIGRE and full-resolution work, rented by the hour | Apple GPUs (Metal) are not supported by either library |
| Excluded | Shift-and-add as a deliverable method | Too limited; may be kept as a debugging aid only |

### 3.1 Compute setup

| Machine | Role | Limits |
|---|---|---|
| Gil's MacBook Pro, Apple M5 Max, 36 GB unified memory | Day-to-day development with Claude Code; CPU tests; LEAP (CPU mode) | No CUDA. FBP works slice by slice (projections ≈ 1.5 GB at full resolution plus one slice), so full-resolution FBP should fit. Iterative methods that keep whole-volume copies in memory are the limiting case: 3072 × 3072 × 300 float32 is 11.3 GB per copy, before slice-count reduction or chunking. Measure, don't assume |
| Rented NVIDIA GPU, 24 GB class (RTX 4090 / L4), on demand | TIGRE, `pytest -m gpu`, full-resolution runs | Paid by the hour; stop when idle |
| Rented A100 80 GB, on demand | Large M4/M5 trade-study sweeps | Paid by the hour |

CPU reconstruction speed on the M5 Max is unknown. Measure it in M0 (runtime of a 256³ OS-SART run) and decide then how much of M3–M5 stays local.

**Open item:** TIGRE's license is not verified. It is believed to be BSD-3. Confirm it, and LEAP's MIT license, against ViBo's commercial plans before building on them (M0).

## 4. Geometry convention and config (proposed; confirm with hardware team)

### 4.1 World frame

- Units: **mm** for lengths, **degrees** for angles in human-facing config. Internal code may use radians.
- Origin: center of the detector front face at its nominal (mid-sweep) position.
- **+z**: normal to the nominal detector plane, pointing toward the source. Reconstructed slices are planes of constant z.
- **+x**: nominal sweep direction of the source.
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

1. **Parametric:** linear or arc trajectory, angular range, number of projections, distances. Expanded to per-projection vectors on load.
2. **Explicit:** per-projection vectors from calibration or encoder logs. This is the Pass B path.

Internally, geometry is always the explicit form. See `configs/geometry_placeholder.yaml`.

### 4.3 Mandatory geometry checks

- Validate on load: `det_u ⊥ det_v`, both unit length, source on the +z side of the detector, no NaNs.
- 3D plot of the trajectory and detector poses (`dts plot-geometry`).
- **Round-trip test:** project a bead phantom with the NumPy reference projector and with each engine adapter. Bead centroids must agree to < 0.1 pixel. Converting general vectors into TIGRE's parameters (per-angle DSO/DSD, offsets, detector rotation, Euler angles) is the highest-risk conversion in the project; this test guards it.

## 5. Simulation scope

**Phantoms**

1. **Geometric:** bead grid at known (x, y, z), thin wire, slab stack at known depths, edge or disc for in-plane MTF, low-contrast discs.
2. **Anatomical (procedural):** a canine-thorax-like phantom (lung, soft tissue, fat, bone ribs and spine) and a limb-like phantom (cortical and trabecular bone, soft tissue). Parameterized by size, from small (cat) to large (equine limb).
3. **Calibration phantom:** bead pattern designed in M5.

**Forward-model effects, added in stages**

| Stage | Effects |
|---|---|
| S0 | Monoenergetic line integrals, no noise |
| S1 | Poisson quantum noise at a set I₀; additive electronic noise |
| S2 | Polyenergetic spectrum → beam hardening (two-material) |
| S3 | Scatter (convolution-kernel model, scatter-to-primary ratio by thickness) |
| S4 | Focal-spot blur; source motion blur during exposure (continuous-motion mode) |
| S5 | Truncation (object wider than the detector's field of view); detector lag (placeholder) |

Pass A results must state which stages were enabled. Simulation fidelity is limited, so image-quality numbers from simulation are **upper bounds**.

## 6. Reconstruction methods

1. **DTS-FBP:** ramp filter × slice-thickness filter × apodization (Hann or similar), then the engine's backprojector. Filter parameters are configurable. Library FDK is not adequate for limited-angle data without this filter design.
2. **OS-SART:** baseline iterative method.
3. **TV-regularized:** ASD-POCS first, then optionally FISTA-TV. Watch for blocky "cartoon" texture on soft tissue.
4. **Truncation handling:** extended reconstruction volume and/or projection extrapolation; masking.

## 7. Metrics

All metrics are computed against ground truth in simulation.

| Metric | Measures |
|---|---|
| Artifact spread function (ASF) | Out-of-plane blur of a bead along z |
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
| M1 | Geometry: schema, loader, validator, plotter, parametric generators (linear, arc), TIGRE and LEAP adapters | Round-trip test < 0.1 px for both engines on linear and arc trajectories |
| M2 | Simulator: phantoms + stages S0–S2 | Simulated projections match reference projector (S0) to numerical precision; noise statistics match theory (S1) |
| M3 | Reconstruction: DTS-FBP, OS-SART, ASD-POCS | Beads reconstruct at correct depths (±1 slice) in all methods |
| M4 | Metrics + comparison report; add S3–S5; trade-study harness | Report comparing methods across ≥ 2 phantoms and ≥ 3 geometries (angular range, n projections, motion mode) |
| M5 | Simulated calibration: phantom design, geometry estimator, sensitivity study | Estimator recovers injected errors within spec; report states the geometry tolerance needed to keep metric loss < X % (X set by Gil) |
| M6 | I/O and tool: mock raw format, dark/flat/log preprocessing, CLI, TIFF/DICOM export, napari viewer | End-to-end run from mock raw files to slice stack by someone other than the developer |
| M7 | Bunker test plan | Ordered acquisition list with predicted results and pass/fail thresholds |
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
3. Per-frame geometry metadata: encoder position and timestamp per projection. Also whether the source moves during exposure.
4. Trajectory type and nominal parameters (section 11).
5. A single owner for the interface spec (`docs/interface_spec.md`). Changes go through that document.

## 11. Placeholder parameters (all to be replaced)

These are rough values of the order typical of radiographic DTS. **They are not ViBo's specifications and have not been checked against any device.**

| Parameter | Placeholder | Real value |
|---|---|---|
| Trajectory | Linear source sweep, stationary detector (also test arc) | TBD |
| Angular range | ±15° (study ±10° to ±30°) | TBD |
| Number of projections | 41 (study 21–81) | TBD |
| Source-to-detector distance at center | 1000 mm (study 1000–1800) | TBD |
| Object-to-detector gap | 20 mm | TBD |
| Detector | 3072 × 3072, 0.139 mm pitch (43 × 43 cm) | TBD |
| Motion mode | Step-and-shoot (also study continuous) | TBD |
| kVp | 70–90 | TBD |
| Patient thickness range | 50–300 mm | TBD |
| Clinical tasks | Thoracic nodules, skeletal fractures, soft-tissue masses | TBD |

## 12. Risks

| Risk | Mitigation |
|---|---|
| Geometry/calibration error dominates image quality (inference) | M5 sensitivity study sets the tolerance early |
| Errors when converting vectors to TIGRE parameters | Round-trip test against reference projector (M1); LEAP's modular-beam input is a direct match to our format, so it is tested first on the Mac |
| Mac CPU too slow for iterative work at useful sizes | Benchmark in M0; move heavy runs to rented GPU |
| Simulation under-models scatter, lag, heel effect | Treat Pass A numbers as upper bounds; report stages used |
| Over-tuning to one phantom | ≥ 2 phantom types; held-out case |
| Interface drift with hardware | Single-owner interface spec |
| License incompatibility with commercial use | Verify in M0 |
| GPU memory limits at full detector size | Detector binning for studies; measure peak memory in M4 |
