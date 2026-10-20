# Real data v1: older multi-emitter DTS system (2018)

First test of the tool chain on measured data: a 45-emitter (7 × 7 minus corners) source array over a
114 × 64 mm, 14-bit detector, hand and finger datasets from 2018. Run on 2026-10-20, Mac M5 Max (CPU),
LEAP. No GPU, no TIGRE.

- **Code:**
  - `src/dtsrecon/io/emitter_array.py`: loader, masks, line integrals, attenuation PNG, MJPEG.
  - `src/dtsrecon/calibration/air.py`: sources from the air images.
  - `src/dtsrecon/calibration/selfcal.py`: anatomy tracking and bundle adjustment.
  - `src/dtsrecon/io/viewer.py`: HTML viewer.
  - `scripts/real_v1.py`: steps `load, air, anatomy, recon, compare, ringsim`, about 12 min in all.
  - Tests: `tests/test_real_data.py` (synthetic inputs).
- **Data:** read in place from the folder named in `configs/real/legacy_emitter_2018.yaml`; nothing is
  copied into the repo.
- **Outputs:** in `data/real_v1/`, which is gitignored (JSON results, geometry YAMLs, TIFF stacks, viewers).
  Figures are in `reports/figures/real_data_v1/`.
- **Simulation stages:** none, except the ring ground-truth check in §5 (S0, analytic, noiseless).

Tags: **[M]** measured from the data; **[I]** inferred (reasoning, not measured); **[A]** assumed (no evidence
either way).

## Summary

1. **The data load and log-transform cleanly** [M]. Andy040618 with its own flats has open-beam residuals of
   0.56 % per pixel and 1.3 % low-frequency (peak-to-peak). Saturation affects 0.014 % of flat pixels, all
   masked. One file is damaged: `Fingers/fingers/B2.tif` (99.9 % of pixels read 65535); it is excluded.
2. **Fingers works with the borrowed Andy040618 flats, with a 5–7× larger flat-field error** [M]: 8.6 %
   low-frequency (up to 22 %) against 1.3 % with matching flats. The exposure differed by a factor of 39 (open-beam
   level 3.67 in p). Usable; the residual shows up as a smooth background error.
3. **The air images give the emitter positions as a regular grid** [M]:
   - pitch 12.73 × 12.67 mm, rotation +0.58°, skew −0.03°;
   - 0.39 mm rms deviation from a perfect grid (max 0.81).

   They give S = 124.8 mm only if the emission is isotropic [M + A]. S trades off against an emission-angle
   term and the air images cannot separate the two. With a cos^m term the fit improves and gives S = 156 mm,
   m = 1.4.
4. **The 1152 air set (rows A–G) is a different setup** [M]: S = 245 mm and pitch 17.5 mm, scaled by 0.72
   against the 2304 set where they overlap. `Andy/wrist` belongs to that setup. The two sets cannot be merged.
5. **The anatomy agrees with the air-fit sources to within tracking noise** [M]. Reprojection is 1.2–1.4 px
   rms at 0.1 mm and improves only 2–6 % when the sources are refined. The refinement moves the sources by
   0.37–0.49 mm rms. Its x-component reproduces between two independent datasets (correlation 0.75), so part
   of it is real, but its effect on the images (≈ 0.1 mm) is below the 0.2 mm voxel. The reconstructions use
   the air geometry.
6. **The depth scale is the main open problem** [M + I]. The Fingers ring reconstructs as an ellipse, 25 mm
   wide in x and 14–15 mm in z (z/x = 0.57–0.62). A simulated circular ring, run through the same geometry and
   the same analysis, gives 0.97–1.13. If the ring is circular, all heights in the air geometry are too small
   by 1.6–2.0× (central 1.75×): S/pitch is about 17, not 9.8. The finger and wrist thicknesses point the same
   way. This conflicts with cmd.txt's SID of 120 mm unless the emitter pitch is about 7 mm, not the 12.7 mm
   seen in the air images.
7. **Reconstructions** (DTS-FBP and OS-SART, 0.2 mm cubic voxels, then 1 mm slices):
   - 35 s and 50 s per dataset;
   - bones, joints and the ring are in focus at distinct depths;
   - viewers: `data/real_v1/{andy040618,fingers}/viewer.html`.
8. **Against the old AVI:**
   - **Banding:** the horizontal banding is replicated detector-fixed structure. Its period grows linearly
     with depth (r = 0.995) from zero at the detector [M]. Ours has 10–30× less banding by the same measure,
     except in the top 1–2 mm of the range [M].
   - **Depth:** a ring-depth match in mm cannot be checked, because the AVI's slice spacing is unknown.
   - **Sharpness:** no robust like-for-like number is possible from an 8-bit, edge-enhanced MJPEG [M].

## 1. Data, loader and log transform

- **Format** [M]:
  - one 16-bit TIFF per emitter ("Teledyne DALSA TIFF image"), file name = emitter label, holding 14-bit data;
  - 2304 × 1300 at 0.0495 mm (114 mm / 2304; the detector size is from cmd.txt);
  - the same detector at 1152 × 650 (2× binned): the same defect sits at full-resolution pixel (740, 480) in
    both;
  - emitter index = row-major position among A2 … G6 (`EMITTERS`).
- **Dark frames:** none were supplied. Raw minima are 0–16 counts under the densest anatomy, so the images
  look offset-corrected. Dark = 0 is used [A] and marked PLACEHOLDER in the config. If there is a residual
  offset d, line integrals are biased by about d/I. For Fingers (open beam about 150 counts) a 2-count offset
  would mean 1.3 % in p.

| | Andy040618 | Fingers | Andy/wrist |
|---|---|---|---|
| views (emitters) | 26 (A2–D7) | 25 (B2 damaged, excluded) | 26 |
| image, pixel | 2304 × 1300, 0.0495 mm | 2304 × 1300 | 1152 × 650, 0.099 mm |
| flats | own (`Andy040618/air`) | **borrowed** (`Andy040618/air`) | own (`Andy/air` = top-level set) |
| projection counts p1 / median / max | 168 / 1004 / 16383 | 30 / 128 / 468 | 4 / 32 / 1231 |
| flat counts median / max | 13657 / 16383 | same flats | 879 / 1813 |
| fixed defects (from the flats) | 42 | 42 | 134 (72 dead) |
| saturated: flats / projections | 0.014 % / 0.016 % | 0.015 % / 0 | 0 / 0 |
| open-beam level (should be 0) | −0.012 (−0.015 … −0.006) | **+3.67** (3.55 … 3.77) | −0.005 (−0.046 … +0.021) |
| open-beam pixel noise (p) | 0.56 % | 3.3 % | 1.7 % |
| open-beam low-frequency residual, median (max) | 1.3 % (1.6 %) | **8.6 % (22 %)** | 1.2 % (3.0 %) |

- **Masks and processing** [M]:
  - Saturated pixels: at most 1335 per flat, clustered at the illumination peak. They are filled from their
    neighbours with the M6 defect filler, as are the fixed defects.
  - Exposure difference: each view's open-beam level is subtracted. That is about 1 % for Andy040618. For
    Fingers it is a factor of 39, so either the exposure or the detector gain differed.
  - Andy/wrist: 26k pixels fall below 10⁻⁴ of the flat, clipped at p = 9.2; these are dark forearm regions at
    the low 1152 exposure.
  - Fingers' 22 % worst case is view A3. The flat mismatch is smooth, consistent with a different
    illumination pattern [I]. Figures: `load_*.png`.
- **Precomputed attenuation PNGs** [M]:
  - The 2304 PNG is the TIFFs with no dark, no masking and negative values clipped to 0: A = ⌊s · max(−ln(I/air), 0)⌋,
    with s = 12619.5 for Andy040618 and 9276.3 for Andy. The fit rms is 0.29 units, which is pure quantization;
    s maps 65535 to p = 5.19 and 7.06 respectively.
  - The 1152 and 576 PNGs of Andy040618 are 2×/4× averages of that log image (0.3–0.5 % rms difference from
    log-of-binned).
  - Andy's 576 PNG differs from log-of-binned by about 0.1 in p; not investigated.
  - `tecplot/*.dat` is the same attenuation as text (A/s to 5·10⁻⁵).

## 2. Geometry from the air images

- **Model** (`calibration/air.py`):
  - I = A · S / r³ (inverse square × obliquity), optionally × (S/r)^m (emission-angle term about the
    detector normal);
  - fitted on ln I of 16×16 (2304) or 8×8 (1152) block means, with saturated and defective pixels excluded;
  - per image, and jointly with one common S.
- **Results** (figures `air_positions.png`, `air_residuals.png`):

| | 2304 set (Andy040618/air, rows A–D) | 1152 set (top level + Andy/air/A3, rows A–G) |
|---|---|---|
| S, joint fit, m = 0 | **124.8 mm** (statistical SE 0.01) | **245.5 mm** (SE 0.05) |
| S, single images, m = 0 | 120.4–128.0 (smaller for edge columns) | 232.6–257.3 |
| joint fit with cos^m | S = 155.8, m = 1.39; residual 0.64 → 0.56 % | runs away (S = 747, m = 22): no constraint |
| single images, m free | S 108–230, m −0.8…+6: degenerate | S 117–880: degenerate |
| grid pitch (columns × rows) | **12.73 × 12.67 mm** | 17.63 × 17.45 mm |
| grid rotation, skew | +0.58°, −0.03° | −0.37°, +0.43° |
| deviation from a regular grid, rms (max) | 0.39 (0.81) mm; 0.26 mm with cos^m | 1.03 (2.36) mm |
| single-image vs joint positions, rms (max) | 1.0 (3.0) mm | 2.4 (6.2) mm |
| fit residual: common to all emitters / emitter-specific | 0.46 % / 0.46 % | 0.46 % / 0.47 % |
| emitter output spread (ln A, rms) | 1.1 % | 1.7 % |

- **What this shows:**
  - **Grid** [M]: the emitters form a regular, unrotated grid. Column 1 is at +x and row A at −y in detector
    coordinates. In the 2304 set the A row is 3 mm beyond the detector edge; in the 1152 set rows A, F, G and
    column 7 are outside it, so those peaks are extrapolated and less reliable [M].
  - **Uncertainty** [M]: the statistical errors (0.02 mm) are negligible. The systematic spread is about
    0.4–1 mm: the single-image and joint fits differ by 1 mm rms, and the regular-grid residual is 0.39 mm.
- **Residual maps** [M]:
  - The common part (0.46 %) is detector structure: a bright strip along the left edge and tile seams. It is
    the detector's own gain pattern, not handled by the model.
  - The emitter-specific part includes a 1–1.5 % dark blob whose position moves with the emitter. It is
    something in the beam near the source, or the emission pattern [I].
- **Overlap of rows A–D** [M]: after a similarity fit the 1152 positions match the 2304 pattern to 0.57 mm
  rms, but scaled by 0.721 and rotated by 1.0°. S/pitch is 9.8 (2304) against 13.9 (1152). That ratio does
  not depend on the assumed pixel size. So this is a different setup (source height and/or emission), not a
  units mix-up, and `Andy/wrist` must not be reconstructed with the Andy040618 geometry.
- **Not identifiable from air images** [M + I]: S and the emission term are degenerate. To leading order the
  image curvature fixes (3 + m)/S² only. A low-frequency gain pattern biases both at the 0.5 % level. The
  positions (peaks) are robust to a symmetric emission term, but would shift if the emitters are tilted or
  aimed.

## 3. Cross-check from the anatomy

- **Method:** the M5 estimator with unknown feature positions (`calibration/selfcal.py`).
  - Shi–Tomasi corners in view C4 (2× binned, 0.099 mm), plus 40 extra corners on the Fingers ring.
  - A plane sweep over parallax for each feature, then a 2D NCC refinement per view.
  - Then a check of consistency with the air-fit sources, and a bundle adjustment of source (x, y) and
    feature (x, y, h).
  - Gauge: the sources' mean and scale are held to the air fit. Anatomy cannot measure them: s → αs + t is
    absorbed by the features. It also cannot measure S, which only relabels heights. Shape and rotation are
    measurable.

| | Andy040618 | Fingers | both together |
|---|---|---|---|
| features tracked / observations used | 149 / 1972 of 3874 | 168 / 2621 of 4200 | — |
| reprojection with air sources, rms (median abs) | 1.40 (0.98) px | 1.20 (0.65) px | 1.29 px |
| after bundle adjustment | 1.31 px | 1.18 px | 1.25 px |
| source correction rms (max); median SE | 0.37 (0.73) mm; 0.12 | 0.49 (1.01) mm; 0.16 | 0.38 mm |
| feature heights p2–p98 (air geometry) | 16.7–32.4 mm | 8.4–18.2 mm; ring 13–18 | — |

- **Consistency** [M]: the anatomy is consistent with the air-fit sources to within tracking noise.
  Refining the sources improves reprojection by only 2–6 %.
- **Correction** [M]: the corrections are 2–3× their standard errors. Between the two datasets they
  correlate 0.75 in x and 0.54 in y, and differ by 0.36 mm rms. So part of the correction, about 0.3 mm and
  mostly in x, is a real shape error of the air fit (e.g. edge columns pulled inward by detector-edge gain)
  [I].
- **Effect on images** [I]: a shadow blur of b × correction ≈ 0.25 × 0.4 ≈ 0.1 mm, below the 0.2 mm voxel.
  The reconstructions therefore use the air geometry. The refined geometries are saved as
  `data/real_v1/geometry_anatomy_*.yaml`. Figures: `anatomy_*.png`.

## 4. Reconstructions

- **Setup:**
  - Detector binned 4× (0.198 mm); volume = detector footprint + 3 mm.
  - 606 × 355 × 323 cubic voxels at 0.198 mm, z = 0–64 mm (cmd.txt's object height). The union of the
    source cones lies inside this.
  - Air geometry (S = 124.8 mm).
  - DTS-FBP: radial filter, 10 mm edge extrapolation. OS-SART: 10 iterations, 6 subsets.
  - Then 1 mm box-averaged slices over the tracked features' height range ± 4 mm: Andy 13–36 mm, Fingers
    5–22 mm.
- **Timings (Mac CPU):**

| | DTS-FBP | OS-SART |
|---|---|---|
| Andy040618 | 35 s | 50 s |
| Fingers | 34 s | 50 s |
| Fingers ring region, full resolution DTS-FBP, 769 × 410 × 606 at 0.0495 mm | 104 s | — |

- **What the stacks show** [M]:
  - Andy040618: carpals and metacarpal bases in focus at 24–28 mm.
  - Fingers: the little finger in focus at 8–11 mm and the other phalanges at 12–16 mm. The ring's lower arc,
    side walls and upper arc are at about 6, 13 and 20 mm.
  - OS-SART has the cleaner background; DTS-FBP has the stronger edge artifacts at the top and bottom of the
    range.
- **Outputs:**
  - `data/real_v1/<dataset>/{dts-fbp,os-sart}_{0.198mm,1mm}.tif` (with JSON sidecars);
  - `viewer.html` (1 mm slices, both methods side by side);
  - `fingers/dts-fbp_ring_roi_full_res.tif`.
  - Figures: `recon_*.png`.
- **Depth labels:** the heights in all stacks are in the air geometry's scale; see §5.

## 5. Depth scale from the ring

- **Real ring** [M]: in the x–z section (maximum over the ring's rows) the ring is an ellipse. Ridge points
  were fitted with an axis-aligned ellipse (`compare_fingers.png`):

| | x semi-axis | z semi-axis | k = z/x |
|---|---|---|---|
| OS-SART, 0.198 mm | 12.5 mm | 7.1 mm | **0.57** |
| DTS-FBP, 0.198 mm | 12.2 mm | 7.6 mm | **0.62** |
| DTS-FBP, full resolution | 12.2 mm | 7.0 mm | **0.57** |
| simulated circular ring (mid radius 11.75 mm), same geometry and analysis: DTS-FBP / OS-SART | 12.0 / 11.4 | 11.7 / 12.9 | **0.97 / 1.13** |

- **Simulated ring:** a steel band of outer radius 12.5 mm and wall 1.5 mm, tilted 13° (as measured in the
  y–z section), on a soft-tissue finger with a bone.
- **What the ground truth shows** [M]: the measurement can be off by about ±15 % between methods, but it is
  not biased toward 0.6. Reconstructing the real data with S scaled by 1/k gives k = 1.04 [M]. That is
  expected, because changing S only relabels heights (z' = z·S'/S), so it checks the analysis rather than the
  data.
- **Interpretation** [I]: if the ring is circular, which is likely for a metal ring, true heights are larger
  than the air-geometry heights by 1/0.64 to 1/0.50, i.e. 1.6–2.0× (about 1.75×). Equivalently S/pitch ≈ 16–20,
  not 9.8:
  - S ≈ 200–250 mm with the air-image pitch (12.7 mm); or
  - pitch ≈ 6–8 mm with cmd.txt's S = 120 mm.
- **Supporting evidence** [I]: with the air geometry, the finger features span 9.8 mm in height and the ring
  is only 14 mm tall around a finger. With ×1.75 these become 17 mm and 25 mm, which are anatomically
  plausible. The wrist region goes from 16 mm to 27 mm thick.
- **Air-fit consistency** [I]: the cos^m air fit moves S in the same direction (156 mm). The single-image
  free-m fits of columns 5–7 give S ≈ 190–230 mm. Both are consistent with emission narrower than isotropic.

## 6. Comparison with the old reconstruction (AVI)

- **The AVI** [M]: 16 MJPEG frames, 4608 × 2600 RGB, 8-bit, decoded without a video library. It is 2× the
  detector grid; the old voxel grid = detector grid [A].
- **Display** [M]: the ring appears as an outline (rims and walls) with a dark interior, so the display is
  edge-enhanced. Our first reading of the outline as a geometry error was wrong: it is the 3 mm band's two
  rims.
- **Ring in focus:**
  - AVI frame 7 of 0–15 [M]. If the frames span 0–64 mm like cmd.txt's Andy run, that is about 30 mm [A].
  - Ours: 13.5 mm in the air geometry, about 24 mm with the ring-based scale.
  - The old recon used its own geometry (keys.txt, missing), so the absolute depths cannot be compared. The
    ordering does agree: the ring and the adjacent phalanges are in focus together in both (ours: ring 12.8 mm,
    phalanges 13.8 mm).
  - The AVI's bone-focus curve is flat and is dominated by streaks in frames 12–15, so it cannot be used [M].
- **Sharpness:** no robust number.
  - The 10–90 % width of the ring rim came out at 0.39 mm for ours (full resolution, object plane) and
    0.13–0.53 mm for the AVI, depending on where the profile is taken.
  - The AVI's edge enhancement and JPEG compression make an edge width there not comparable [M].
  - By eye, ours shows the ring band and the bone cortex without the AVI's vertical streaking
    (`compare_fingers.png`).
- **Horizontal banding** (`compare_banding.png`):
  - **Size** [M]: the banding index (rms of the detrended row means over that of the column means, palm
    region) is 1.2–4.7 in the AVI, against 0.12–0.19 for OS-SART and 0.18–0.27 for DTS-FBP over the object
    range. The top 1–2 slices of our range reach 0.7 (OS-SART) and 1.5 (DTS-FBP), from the detector edge.
  - **Period** [M]: the AVI band period grows linearly with frame, 0.180 mm per frame (r = 0.995), and
    extrapolates to zero at frame −0.7, i.e. at the detector.
  - **Mechanism** [I]: horizontal structure fixed to the detector (the bottom edge, or a flat-field
    mismatch) backprojected once from each source row lands at spacing (row pitch) × z/S. Hence the period
    grows with depth.
  - **Why ours is cleaner** [I]: our flat-field and defect handling, the edge extrapolation and OS-SART all
    suppress it.

## 7. What these data cannot tell us

- **Absolute source distance and emitter pitch:** only their ratio is constrained, by the ring (§5); the air
  images cannot separate S from the emission pattern (§2), and the anatomy cannot measure scale or S (§3).
  Needed: a physical measurement (emitter pitch, source-to-detector distance), or a phantom of known size and
  height (e.g. the M5 two-plane bead phantom).
- **Dark offset:** no dark frames, so the dark level is assumed to be 0.
- **Fingers' own flat field:** no flats of its own; the borrowed ones leave an 8.6 % smooth background
  error.
- **Linearity and lag:** detector linearity, lag between emitters, and scatter cannot be checked without
  dedicated frames (darks, flats at several exposures, a step wedge).
- **The old system's geometry and slice spacing:** keys.txt is missing, and so is the Fingers recon
  command. The AVI's depth axis therefore cannot be mapped to mm, and the old result cannot be reproduced.
- **Density accuracy:** no reference object of known attenuation.
- **The 1152 setup's geometry beyond the air fit:** Andy/wrist was not reconstructed. It needs its own
  geometry and has the same depth-scale ambiguity.

## 8. Questions and next steps

1. **The emitter pitch of the source array and the source-to-detector distance for each session.** This is
   the one number that fixes the depth scale; is cmd.txt's 120 mm the source plane, the target, or something
   else?
2. **Can keys.txt, or the command used for Fingers, be found?** It would give the old geometry and the AVI's
   slice spacing.
3. **If this system is imaged again, add:**
   - a few darks;
   - flats at the same settings as each scan;
   - one scan of a known-size object at a known height (two-plane bead phantom, or a ruler and a coin on
     spacers).
4. **On our side, once the scale is known:**
   - rerun `scripts/real_v1.py` with the corrected S;
   - consider a joint air + anatomy + ring fit;
   - reconstruct Andy/wrist with the 1152 geometry.
