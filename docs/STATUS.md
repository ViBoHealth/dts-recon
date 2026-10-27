# Status (2026-10-20)

Pass A (simulation). All physics, detector and geometry values for the new system are placeholders. A first
real-data test on an older 45-emitter system (2018 hand data) is in `reports/real_data_v1.md` (item 28); the second,
with the old software's keys files, in `reports/real_data_v2.md` (item 29).
Details and numbers are in `docs/decisions.md` (item numbers below).

Everything below ran on the Mac (M5 Max, CPU, LEAP). TIGRE and the GPU machine have not been used.

## Milestones (brief §8)

| | Built | Acceptance (brief §8) | Open |
|---|---|---|---|
| M0 | Repo, LEAP CPU build (two upstream bugs patched), Mac benchmark | Mac side met | GPU side (TIGRE + LEAP on a rented GPU), TIGRE license check, CI (no remote yet) |
| M1 | Geometry schema, loader, validator, plotter, grid / linear / arc generators, explicit geometry, LEAP adapter | Round-trip < 0.1 px **for LEAP** | TIGRE adapter and its round-trip test (GPU) |
| M2 | Analytic phantoms (IQ, dental, calibration), stages S0–S2 | Met (tests) | — |
| M3 | DTS-FBP for 2D grids, OS-SART, ASD-POCS | Met: beads within ±1 slice in all methods | — |
| M4 | Metrics, stages S3–S5, trade study (8 geometries × 3 methods × 2 phantoms), motion, truncation, slice-thickness filter, compute | Met, with no written report (Gil: none during early testing) | Dose-matched rerun deferred to grid optimization. RMSE/SSIM uninformative (item 23) |
| M5 | Two-plane bead phantom, geometry estimator, sensitivity study, closed loop | Met. Limits set by Gil (2026-10-10): ≤ 10 % blur, ≤ 0.5 mm distortion (item 27) | Revisit once stage repeatability is known |
| M6 | Raw format, mock detector, preprocessing, `dts` CLI, TIFF / DICOM export, napari viewer, user guide | End-to-end run works (item 26); **not yet run by someone other than the developer** | That run (`docs/USAGE.md`) |
| M7 | — | — | Paused: bunker not available |
| M8 | — | — | Optional, not started |

## Main results so far (simulation, placeholder physics)

- **Geometry choice.** A 400 mm source extent beats 300 mm, and gains level off beyond about 25 projections
  (item 22). The fully sampled height at 6×6 / 400 / SID 500 is about 167 mm.
- **Methods.** Depth resolution, best to worst: ASD-POCS (2.3–5 mm ASF), then OS-SART, then DTS-FBP.
  - DTS-FBP misses wide, low-contrast objects (items 22–23).
  - ASD-POCS was tuned on the IQ phantom at the baseline geometry, which gives it a home advantage there.
  - A method should not be chosen without clinical image review (brief §7).
- **Calibration.** Calibrating gets sources to ≤ 0.1 mm, from errors of several mm.
  - The limit is how accurately the phantom's bead positions are known: 0.05 mm bead error gives about
    0.36 mm source error (item 24).
  - Grid rotation and tilt blur the image. Offset, SID and scale errors move targets instead.
- **Tool.** From raw frames to a slice stack on the Mac, at 0.4 mm:
  - OS-SART: 2 min, 5.4 GB peak.
  - ASD-POCS: 3 min, 6.9 GB peak.
  - Calibration from a raw scan: 5 s.

## First real data (older 45-emitter system, 2018; item 28)

- **Tools:** loader, masks and log transform for the old format; sources from air images; anatomy cross-check
  (M5 estimator with unknown points); DTS-FBP / OS-SART stacks with an HTML viewer. All run on the Mac in
  about 12 min.
- **Geometry:** emitters on a regular 12.7 mm grid (0.4 mm rms off-grid). The anatomy agrees with it to within
  tracking noise.
- **Depth scale:** not fixed by these data. Air images give S = 125 mm only for isotropic emission. A circular
  ring reconstructs 1.6–2.0x too flat (simulation-checked), so true heights are probably about 1.75x larger.
- **Against the old system's AVI:** its horizontal banding is replicated detector-fixed structure (period ∝ depth).
  Ours has 10–30x less.
- **Answered by Gil (2026-10-27):** pitch 10 mm; no distances on record; keys.txt not available. See item 29.

## Real data v2 (keys files, new wrist and set-2 data, old-software slices; item 29)

- **Keys:** exact 10 mm grids. Their orientation on the detector is measured from parallax (identity); the grid shape
  is confirmed to 0.2 mm.
- **Setups:** Fingers and the new wrist share the 1152 air set's setup (B). v1 had given Fingers setup A's geometry.
- **S with the 10 mm pitch:** setup B ≈ 195 mm (ring), setup A ≈ 145 mm (air images, aimed-emission model). Both are
  placeholders. Setup B's air images do not support the ring value (unresolved).
- **Recon:** the new wrist and set 2 are reconstructed (DTS-FBP, OS-SART). The old dental and fish slices were reviewed
  qualitatively: replicated detector-fixed lines, spacing linear in slice index.
- **Gil's answers (2026-10-27):** pitch is 10 mm. No distances on record and no further files, so S stays inferred and
  the dental and fish comparison stays qualitative.

## Known limitations

- **LEAP only.** There is no second engine to cross-check against. The NumPy reference projector covers the
  geometry, but not engine-specific behaviour at full size.
- **ASD-POCS lesion depth is fragile.** For the faint dental lesion it moves by about 1 mm from float32 vs
  float64 alone, and by up to 5.7 mm across code versions (item 26).
- **Truncation.** Iterative methods need the whole object inside the volume. `dts recon` adds a 20 mm margin
  by default (item 23).
- **Step-and-shoot only.** Continuous motion is modelled but out of scope (Gil, 2026-10-10).

## Waiting on Gil

- Someone to do the M6 acceptance run.
- When to run the GPU session (TIGRE, GPU tests, M0/M1 remainder) and the TIGRE license check.
- Whether to report the two LEAP CPU bugs upstream (decisions items 2 and 6).

## Waiting on the hardware team (brief §10)

- **Raw format:** file type, header, bit depth, byte order. Strawman: `docs/raw_format.md`.
- **Dark and flat procedure:** how they are acquired and how often. Exposures per position.
- **Encoder axes:** the frame they report in. Stage repeatability, which calibration assumes.
- **Detector characterization:** gain, noise and defects. Source spot size and kVp for dental.
- **Calibration phantom:** can it be manufactured with bead positions measured to about 0.02 mm (CMM)?
