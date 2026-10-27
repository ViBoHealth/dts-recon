# Real data v2: keys files, emitter pitch, source distance, new wrist and set-2 data, old-software slices

Second real-data pass on the older multi-emitter system (2018). New: four WeTransfer folders with the old software's
keys files (emitter positions), a 2304 × 1300 wrist scan, 45 attenuation PNGs of a smooth object, and two
old-software slice stacks (dental, fish). Run on 2026-10-27, Mac M5 Max (CPU), LEAP. No GPU, no TIGRE.

- **Code:**
  - `src/dtsrecon/io/emitter_array.py`: `read_keys`, `keys_rc` (both label styles).
  - `src/dtsrecon/calibration/selfcal.py`: geometry-free parallax: `track_free` (NCC, optionally chained over grid
    neighbours), `grid_chain`, `factor_parallax` (robust rank-1), `compare_pattern` (all eight axis orientations).
  - `src/dtsrecon/calibration/air.py`: `fit_joint_grid` (sources fixed to a keys grid; emission optionally aimed),
    `fit_grid_multistart`, `model_maps`.
  - `scripts/real_v2.py`: steps `keys, load, parallax, air, ring, depth, recon, old`, about 30 min in all.
  - Tests: `tests/test_real_data.py` (synthetic inputs; 3 new tests).
- **Data:** read in place: v1 folder (`/Users/gtravish/Downloads/sendgb-H3vf2HxCCaz`) and the four `data/wetransfer_*`
  folders (gitignored). Configs: one per setup in `configs/real/legacy_2018_*.yaml`.
- **Outputs:** `data/real_v2/` (JSON, logs, stacks, viewers; gitignored). Figures: `reports/figures/real_data_v2/`.
- **Simulation stages:** S0 only (analytic, noiseless), for the simulated ring in §5.1.

Tags: **[M]** measured from the data; **[I]** inferred (reasoning, not measured); **[A]** assumed.

Names used below: **setup A** is the session of `Andy040618/air` (2304 air set, rows A–D); **setup B** is the session of
the 1152 air set (rows A–G). §2 shows which data belong to which.

## Summary

1. **The keys files are exact 10.000 mm grids** [M]. The three A–G files differ only by an offset (x 85.7/85.7/88.2,
   y 2.5/3.0/5.0 mm). The fish file is a full 7 × 7 grid with different labels, and its x runs the other way.
2. **Orientation: keys x lies along the detector columns and keys y along the rows, both increasing the same way
   ("identity")** [M]. This was tested from parallax without assuming any source pattern:
   - Fingers, Andy040618 and the wrist give the same answer.
   - A regular 10 mm grid fits the recovered source pattern to 0.21, 0.27 and 0.55 mm rms.
   - Every other orientation is ≥ 18.7 mm rms off. The point-reflected one is excluded because it would put every
     feature below the detector.
   - Rotation is below 0.45°. The x/y pitch ratio is 1.007 and 1.008 (Fingers, Andy040618) and 1.025 (wrist).
   - **Parallax cannot measure the absolute pitch** (gauge). It confirms the grid's shape and orientation. The 10 mm
     itself is confirmed by Gil (2026-10-27).
3. **Fingers belongs to setup B, not setup A** [M]:
   - With the 1152 air set as flats its open-beam residual is 1.27 %, against 8.6 % with the 2304 air set v1 used.
   - Its exposure offset is the same as the new wrist's (+1.126 in p).
   - The new wrist is setup B too (2.2 % against 7.2 %).
   - So v1 reconstructed Fingers with the wrong setup's geometry, which accounts for part of v1's flat ring.
4. **12.7 mm against 10 mm** [M + I; pitch confirmed 10 mm by Gil]:
   - The air-image peaks sit on a 12.70 mm grid in setup A and a 17.5 mm grid in setup B.
   - Sources on a 10 mm grid with emission along the detector normal fit the air images 3× worse
     (1.99 % against 0.66 % rms in setup A) [M].
   - With emission aimed outward (axes diverging from the array) the 10 mm grid fits as well as free positions:
     0.570 % against 0.560 % [M]. This is the new air-fit model.
   - Gil confirms the pitch is 10 mm, so the ½-inch / units reading of the 12.7 mm is ruled out. The air-image peaks
     are displaced outward from the sources in both setups, by 27 % (A) and 75 % (B) of each source's offset.
     Aimed (tilted) emission is the one tested model that does this; it fits setup A, not setup B (§5.3).
5. **Source distance S, with the 10 mm pitch.** The estimates come from different setups:

   | Setup | Method | S |
   |---|---|---|
   | B | Fingers ring, by matching a simulated circular ring [M + A: the ring is circular] | 185 mm (DTS-FBP), 208 mm (OS-SART); about 195 ± 15 |
   | A | Air images with the pitch fixed: profile minimum [M, model-dependent] | 145 mm (+25/−10) |
   | B | Air images | no usable estimate: the fit needs the grid 20–40 mm away from the keys position [M] |
   | any | Parallax heights and object thickness [I] | plausibility only |

   - cmd.txt's 120 mm is for setup A, about 15 % below the air estimate there.
   - So no setup has two independent estimates to compare.
6. **Reconstructions** [M]: the wrist (set 1, setup B) and set 2 were reconstructed with DTS-FBP and OS-SART at
   S = 195 mm, a placeholder. Stacks and viewers are in `data/real_v2/{wrist_v2,set2}/`. In the wrist, the pisiform,
   the carpals and the distal radius/ulna are in focus at distinct depths. Set 2 has no internal structure to focus.
7. **Old-software slices** [M]:
   - Both stacks show replicated detector-fixed lines: 7 copies, one per emitter row (fish) or column (dental). Their
     spacing grows linearly with slice index: 1.61 px/slice for dental, 0.40 px/slice for fish.
   - It extrapolates to zero at slice 0.94 (dental) and −0.3 (fish), so the slices start at or just above the detector.
   - Spacing per slice = pitch × Δz / (S × slice pixel), so Δz needs S and the pixel size, which are unknown.

## 1. Keys files

| | file | emitters | column step | row step | x range | y range | centre − (57, 32) |
|---|---|---|---|---|---|---|---|
| set 1 (wrist) | `keys_all.txt` | 45 (A2…G6) | (−10, 0) | (0, +10) | 25.7–85.7 | 2.5–62.5 | (−1.3, +0.5) |
| set 2 | `keys.txt` | 45 | (−10, 0) | (0, +10) | 25.7–85.7 | 3.0–63.0 | (−1.3, +1.0) |
| set 3 (dental) | `keys.txt` | 45 | (−10, 0) | (0, +10) | 28.2–88.2 | 5.0–65.0 | (+1.2, +3.0) |
| set 4 (fish) | `keys.txt` | 49 (atten1RRCCC, 7 × 7) | (+10, 0) | (0, +10) | 40–100 | 30–90 | centre (70, 60) |

- **Grid** [M]: all four are exact grids (rms off-grid 10⁻¹⁴ mm) at 10.000 mm, unrotated. The A–G files give column 1
  the largest x, so x falls with the column number. In the fish file x rises with CCC.
- **Units:** millimetres. The files don't say, but Gil confirms the 10 mm pitch (2026-10-27). The A–G ranges (x 25.7–88.2, y 2.5–65) fit a frame whose
  origin is a corner of the 114 × 64 mm detector, with the grid roughly centred on it. That is what the configs
  assume (`origin_mm: [57, 32]` = detector centre, PLACEHOLDER).
- **Fish** [I]: its centre (70, 60) and 7 × 7 layout do not fit the 114 × 64 detector, so it is probably another
  detector and/or array.
- **No S in any keys file**, as stated in the task.

## 2. Which data belong to which setup (flat-field match)

Each data set was flat-fielded with each air set. The low-frequency residual is the peak-to-peak of a quadratic fitted
to the open-beam pixels, after the per-view level is removed (v1 method).

| data | flats: 1152 air set (setup B, 2× upsampled) | flats: Andy040618/air (setup A) |
|---|---|---|
| Andy040618 (26 views) | level −2.735; residual 7.28 % (max 11.6) | level −0.012; **1.26 %** (max 1.65) |
| Fingers (25 views) | level +1.126; **1.27 %** (max 2.66) | level +3.674; 8.60 % (max 21.7) |
| wrist, set 1 (44 views) | level +1.126; **2.18 %** (max 2.59) | level +3.632 (25 views); 7.20 % (max 9.42) |

- **Assignment** [M]: Fingers and the new wrist both match setup B as well as Andy040618 matches its own flats.
  - The illumination pattern encodes the source positions and distance, so a match over all views means the same
    source geometry [I].
  - Both have the same exposure offset (+1.126), so they were taken with the same exposure, probably in one session [I].
  - The flats for both come from the 1152 air set, upsampled 2×. Its pixel-level structure therefore does not cancel
    (pixel std 3.4–3.5 %), but the low-frequency field does.
- **Consequence for v1** [M + I]: v1 reconstructed and ring-analysed Fingers with setup A's air geometry
  (S/pitch 9.8). Its depth scale was therefore wrong for a reason v1 could not see. §5.1 redoes the ring in setup B.
- **Set 2** [M]:
  - These are attenuation PNGs, so no flats can be tested.
  - Noise grows exponentially with the PNG value, so the encoding is logarithmic like the Andy PNGs.
  - The open-beam level is +0.225 in p at Andy040618's scale (0.18–0.50 per view), so its flats did not quite match.
    The level is subtracted per view.
- **Set 2 scale s (A = s·p)** [M]: not recoverable.
  - The noise-growth fit gives 9084. The same fit on PNGs of known scale gives 9809 for s = 12619.5 (−22 %) and
    10672 for s = 9276.3 (+15 %).
  - The old PNGs are not normalised to their maximum either: Andy040618's maximum is 58930, Andy's 1152 PNGs' is
    65535, set 2's is 65534.
  - Set 2 is therefore reconstructed at s = 12619.5 [A]. That scales the attenuation values only, not the geometry.

Figure: `load_png_noise.png`.

## 3. Orientation and grid shape from parallax

- **Method** (`selfcal.track_free`, `factor_parallax`, `compare_pattern`):
  - Corners are taken in the reference view. Each is tracked by NCC from view to neighbouring view over the label grid
    (A1–G7 adjacency only, no positions).
  - With all sources at one height, the shadow displacements are exactly d_ij = −b_j (s_i − s_ref). The views ×
    features matrix is rank 1, so a robust factorisation gives the source pattern up to one scale and a translation.
    Features with inconsistent motion (> 2× the median residual) are trimmed.
  - The pattern is then fitted to the keys in all eight axis orientations (swaps and flips) with a common scale.
  - b_j > 0 (features between detector and sources) fixes the sign, so the point reflection ("flip x+y") would need
    every feature below the detector.
  - Andy040618 had too few clean chained tracks, so its plane sweep was seeded with the Fingers pattern (data, not keys).
  - Set 2 has no point features (a smooth object, §6). Its windows on the outline's edge strength are a weak test.
- **Test** on synthetic blobs (`test_parallax_recovers_orientation_without_a_pattern`): identity recovered, b
  correlation > 0.999, wrong orientations > 3 mm off.

| | Fingers (B) | wrist, set 1 (B) | Andy040618 (A) | set 2 |
|---|---|---|---|---|
| features / observations | 48 / 973 | 64 / 1926 | 234 / 3760 | 113 windows / 4424 |
| rank-1 residual rms (median), 0.198 mm px | 0.63 (0.33) | 1.59 (0.87) | 1.46 (1.08) | 22.2 (5.3) |
| **identity**: rms vs 10 mm grid; pattern explained | **0.21 mm; 99.99 %** | **0.55 mm; 99.96 %** | **0.27 mm; 99.98 %** | 12.3 mm; 79 % |
| best other orientation (not the point reflection) | flip x/y: 18.9 mm | swap: 26.4 mm | flip x/y: 18.7 mm | swap+flip: 26.6 mm |
| point reflection | needs b < 0 | needs b < 0 | needs b < 0 | needs b < 0 |
| rotation vs keys | +0.01° | +0.13° | +0.44° | (+5.9°) |
| affine: x/y pitch ratio, skew | 1.007, −0.18° | 1.025, −0.78° | 1.008, +0.15° | (0.92, −9.5°) |
| affine residual rms | 0.20 mm | 0.40 mm | 0.19 mm | (11.8 mm) |

- **Orientation** [M]: identity in all three anatomy data sets, by a margin of ~90× in rms.
  - Mapping: keys x maps to detector x (columns), keys y to detector y (rows). Column 1 at +x and row A at −y, as v1's
    air fit found.
  - Set 2 agrees, but only weakly: 12 mm against ≥ 26.6 mm.
- **Grid shape** [M]:
  - The 10 mm grid's shape is confirmed to 0.2 mm rms in Fingers and Andy040618.
  - The wrist is noisier: 0.4 mm, x/y 1.025, and a +0.6 mm row-D offset.
  - The row-D offset is the same with reference views D4, C4, E4 and D3, so it is not the reference. A per-view
    translation term (object motion) does not reduce the residual (1.59 → 1.59 px).
  - Fingers, in the same setup, shows no row-D offset (−0.03 mm). So it is specific to the wrist data; the cause is
    unknown [I].
- **Scale** [M + I]: parallax measures b_j × pitch only. The 10 mm cannot be tested this way (Gil confirms it), nor
  can the keys offsets (translation is also a gauge freedom).
- **Rotation** [M]: Andy040618's +0.44° agrees with setup A's air fit (+0.64°). In setup B the parallax rotation
  (+0.01°, +0.13°) disagrees with the air-peak grid's (−0.68°), so the peaks are not at the sources there [I].

Figure: `parallax_patterns.png` (affine residual per emitter, ×10).

## 4. 12.7 mm (air images) against 10 mm (keys)

- **Model.** The source positions are held on the keys grid: s_i = c + scale · R(θ) (k_i − k̄).
  - The intensity is I = A_i · S / r³ · cos(ψ)^m.
  - ψ is the angle between the ray and the emission axis. The axis points from source i to
    q_i = P + (1 + κ)(s_i − P) on the detector: κ > 0 means the axes diverge from the array; κ = 0 means all axes lie
    along the detector normal.
  - Test: `test_pitch_fixed_air_fit_needs_aim` recovers S, m, κ from synthetic aimed images. A normal-emission fit of
    the same images leaves a residual, and a free-scale isotropic fit puts the sources on the wider peak grid.

| fit (rms of ln I) | setup A (2304 air set) | setup B (1152 air set) |
|---|---|---|
| free positions, isotropic (v1) | 0.640 %, S 124.8 | 0.657 %, S 245.5 |
| free positions, cos^m about the normal (v1) | 0.560 %, S 156, m 1.4 | 0.641 %, runs away (S 747, m 22.5) |
| grid, free pitch, isotropic | 0.663 %: **pitch × 1.270**, θ +0.60° | 0.668 %: **pitch × 1.754**, θ −0.68° |
| grid, free pitch, cos^m | 0.574 %: pitch × 1.276, S 157, m 1.5 | 0.649 %: pitch × 1.78, runs away (S 600) |
| **10 mm grid, isotropic** | **1.988 %** (S 116) | **1.636 %** (S 202) |
| **10 mm grid, cos^m about the normal** | **1.962 %** (S 135, m 0.9) | **1.633 %** (S 246, m 1.4) |
| **10 mm grid, cos^m aimed (κ free)** | **0.570 %**: S 143, m 0.66, κ 1.34 | 0.636 %, runs away (S 651, m 17; grid centre ~120 mm off) |

- **The emission-angle term alone cannot reconcile them** [M]. A symmetric term about the normal leaves every peak at
  its source's foot, so with 10 mm sources the residual stays 3× too large in both setups.
- **Emitter tilt (aimed emission) can** [M]. With axes diverging from the array, the 10 mm grid fits setup A as well
  as v1's free fit. In setup B it fits as well as free positions, but S runs away (§5.3). Setup A's best fit has
  m ≈ 0.5–0.7 and κ ≈ 1.3–1.6, i.e. the edge columns' axes are tilted ~16–19° outward. At larger S the profile
  trades this for a sharper beam and less tilt (S 200: m 3.8, κ 0.48, ~4°).
- **Something else: units** — ruled out. The setup-A peak grid is 12.73 × 12.67 mm (12.70 mm = ½ inch), but Gil
  confirms a 10 mm pitch (2026-10-27).
  - So the peaks are displaced from the sources in both setups. In setup B the parallax rotation also disagrees with
    the air-peak rotation (§3).
  - Setup A alone is consistent with axes diverging from a point ~90–120 mm behind the array (S/κ near the profile
    minimum) [I]. Setup B does not fit that picture with the sources at the keys position (§5.3), so it is not a
    tested physical model.
- **Decision:**
  - Reconstructions use the keys grid (10 mm, confirmed), never the air-peak grid.
  - The air-fit model has the aimed term (`fit_joint_grid(..., aim=True)`). It is adequate for setup A, not for setup
    B.
  - Air-image peak positions must not be used as source positions for this system.

## 5. Source distance S (pitch fixed at 10 mm)

### 5.1 From the Fingers ring (setup B)

- **Setup:** Fingers with setup-B flats and the keys grid (`keys_all.txt`, identity), reconstructed at S = 150, 197
  and 265 mm. Detector 4× binned, voxels 0.198 mm. The ring's x–z ellipse is measured as in v1, with one ROI shared by
  both methods (OS-SART's own ROI caught only the bright end of the ring).
- **Bias check:** at each S, a simulated circular steel ring (outer radius 12.5 mm, wall 1.5, tilted 13°, on a finger)
  is placed at the measured centre, projected (S0) and measured the same way.
- **Estimate:** S is where k_real = k_sim, interpolated.

| S (mm) | DTS-FBP: k real / k sim | OS-SART: k real / k sim | ring centre height zc (FBP, OS-SART) |
|---|---|---|---|
| 150 | 0.922 / 1.209 | 0.731 / 1.289 | 17.6, 18.0 mm |
| 196.7 | 1.165 / 1.068 | 0.962 / 1.027 | 22.8, 23.7 mm |
| 264.7 | 1.437 / 1.043 | 1.269 / 0.931 | 30.8, 32.1 mm |
| **S where k_real = k_sim** | **184.9 mm** | **207.7 mm** | |

- **Result** [M + A]: S_B = 185–208 mm, about 195 ± 15 mm (the half-range between methods). That is S/pitch ≈ 19.5.
- **Assumption** [A]: the ring is circular. A 5 % oval would move S by about 5 %.
- **Behaviour with S** [M]: the real ring's k grows roughly in proportion to S, as expected (heights scale with S at
  fixed source x, y). The simulated ring's k falls toward 1 as S grows: the bias is smaller at smaller tomographic
  angles here. This is why S had to be found by iterating rather than by one rescaling.
- **Against v1** [I]: v1's "S/pitch ≈ 17" came from setup A's geometry. The ~170 mm expected from it is superseded by
  this setup-B value.

Figure: `ring_S.png`.

### 5.2 From parallax heights and thickness (plausibility only)

Feature heights h = S·b/(1 + b), with b measured at the keys pitch [M], p2–p98:

| S | Fingers (B) | wrist (B) | Andy040618 (A) | set 2 |
|---|---|---|---|---|
| 120 | 10.6–17.7 | 15.1–22.2 | 19.3–36.5 | 18.0–25.9 |
| 140 | 12.4–20.7 | 17.6–25.9 | 22.6–42.6 | 21.0–30.2 |
| 170 | 15.0–25.1 | 21.4–31.4 | 27.4–51.8 | 25.6–36.7 |
| 200 | 17.6–29.5 | 25.1–36.9 | 32.2–60.9 | 30.1–43.2 |
| 245 | 21.6–36.2 | 30.8–45.2 | 39.5–74.6 | 36.8–52.9 |

- **What this can say** [I]: only plausibility, and it needs the distance from the detector's sensor plane to its
  outer surface (unknown).
  - Andy040618 (setup A): features 20 mm deep at S = 140–145, starting 23 mm above the sensor. That is plausible for a hand
    on a cover, and it would favour the lower end of §5.3's range.
  - Fingers at S = 195: features at 17–29 mm, so the finger's underside is ~12 mm above the sensor plane.
- **Ring contact** [M + I]: the ring's centre height is zc = 0.118·S (0.116–0.121, both methods, all S). Its lowest
  point (outer radius ≈ 12.5 mm) is at 0.118·S − 12.5:
  - 0 mm at S ≈ 106;
  - 5 mm at S ≈ 148;
  - 10.5 mm at S = 195.
  If the ring rested on the detector cover, S = 195 implies ~10 mm from sensor to cover top (or a support or raised
  hand). The ring's 13° tilt says the finger was not flat, so this is a check, not an estimate.

### 5.3 From the air images with the pitch fixed

S profile with the 10 mm grid and aimed emission (`fit_grid_multistart`, five starts per S, m ≥ 0). "Gain-corrected"
means the detector's common residual map (v1's 0.46 % gain pattern) is divided out first.

| S (mm) | 100 | 120 | 130 | 135 | 140 | 145 | 150 | 160 | 180 | 200 | 240 | 280 | 340 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| setup A, raw rms % | 4.65 | 1.36 | 0.721 | 0.602 | 0.573 | **0.571** | 0.575 | 0.592 | 0.632 | 0.668 | 0.726 | 0.765 | 0.804 |
| setup A, gain-corrected % | 4.62 | 1.27 | 0.542 | 0.377 | 0.329 | **0.321** | 0.326 | 0.347 | 0.399 | 0.448 | 0.521 | 0.572 | 0.621 |
| setup A, m / κ at the profile point | 0 / 1.8 | 0.1 / 3.5 | 0.2 / 2.9 | 0.3 / 2.2 | 0.5 / 1.6 | 0.7 / 1.2 | 1.0 / 1.0 | 1.5 / 0.77 | 2.6 / 0.57 | 3.8 / 0.48 | 6.6 / 0.39 | 10 / 0.36 | 16 / 0.33 |

Setup B (1152 air set, 16 px blocks, two starts per S):

| S (mm) | 140 | 180 | 200 | 220 | 280 | 360 | 480 |
|---|---|---|---|---|---|---|---|
| setup B, raw rms % | 5.91 | 2.12 | 1.45 | 1.01 | 0.538 | 0.534 | 0.532 |
| setup B, gain-corrected % | 5.88 | 2.06 | | 0.895 | 0.314 | 0.308 | 0.309 |
| 10 mm grid centre (keys: −1.3, +0.5) | | | (−2.9, +6.4) | | (−7.8, +21.4) | (−19.7, +40.6) | |

- **Setup A** [M, model-dependent]:
  - The minimum is at S = 145 mm in both versions, flat over 140–150 (m 0.7, κ 1.2). The rms rises 10 % at
    ~133–137 mm below and ~163–178 mm above; 120 mm is 2–4× worse.
  - So S_A = 145 (+25/−10) mm.
  - Fixing the pitch therefore does help, but through the aimed model: the S–m degeneracy becomes a weaker S–(m, κ)
    trade-off, and the data prefer its low-S end.
  - Caveat [I]: a different aiming model (per-emitter tilts, a non-cos^m beam) could move this; the residual is still
    dominated by detector structure.
- **Setup B** [M]:
  - The residual is steep below S ≈ 280 mm and flat above it. Taken at face value that is a lower bound of ~280 mm,
    against the ring's 185–208 mm.
  - It is not usable: the low residual at S ≥ 280 is reached only by moving the 10 mm grid's centre 21–40 mm away from
    the keys position. Near the keys position (S = 200) the fit is poor (1.45 %) with an absurd divergence (κ = 7.5).
  - So the single-centre aimed model does not describe setup B's air images with the sources where the keys put them
    [M]. Possible reasons [I]:
    - per-emitter aiming;
    - the extrapolated peaks: rows A, F, G and column 7 lie off the detector;
    - the keys origin being wrong for this session.
  - This is an open tension with the ring, not a second estimate.
- **cmd.txt's 120 mm** (setup A) is outside the steep side of setup A's profile with this model (1.27–1.36 % against
  0.33–0.57 %). The fit is not compatible with 120 mm unless the emission model is wrong [M + I].

Figure: `air_profile.png`.

### 5.4 Do they agree?

- **Not testable directly** [M]: the ring (setup B) and the air estimate (setup A) are for different setups.
- **Direction agrees** [I]: S_B ≈ 195 > S_A ≈ 145 matches v1's finding that setup B's illumination is flatter (the
  iso-fit S was 245 against 125).
- **Within setup B the air images do not support the ring value** (§5.3). That is unresolved, and it is the main
  reason S_B stays a placeholder.
- **The open number** is the physical source distance per setup. Each estimate above rests on one assumption: a
  circular ring, an emission model, or an unknown cover thickness.

## 6. Reconstructions (set 1 wrist and set 2)

- **Setup:**
  - Keys grid (identity, keys origin at the detector corner [A]); S = 195 mm for both (placeholder: §5.1 for the wrist;
    for set 2 the setup is unknown and its keys offsets are closest to setup B's).
  - Detector binned 4× (0.198 mm); volume = detector footprint + 3 mm, cubic voxels.
  - DTS-FBP (radial filter, 10 mm edge taper) and OS-SART (10 iterations, 6 subsets).
  - Then 1 mm slices over the tracked feature heights ± 6 mm (wrist), or the full range (set 2).
- **Flats:**
  - Wrist: setup B's 1152 air set, upsampled 2× (residual 2.2 %, §2).
  - Set 2: its own PNG attenuation, open-beam level removed, at s = 12619.5 [A].

| | volume | DTS-FBP | OS-SART | 1 mm slices |
|---|---|---|---|---|
| wrist (44 views) | 606 × 355 × 308 | 63 s | 90 s | 18.9–40.7 mm |
| set 2 (45 views) | 606 × 355 × 339 | 74 s | 101 s | 0.5–65.8 mm |

- **Wrist** [M]:
  - In focus, in order of height: the pisiform at ~26 mm, the carpal rows at 29–32 mm, the distal radius and ulna at
    32–36 mm (heights at S = 195; they scale with S).
  - OS-SART has the cleaner background. DTS-FBP has stronger edge artifacts at the top and bottom of the range.
  - The 2× upsampled flats leave fine fixed-pattern noise, so this stack is noisier than Andy040618's.
- **Set 2** [M]: a smooth object with no internal structure. Its outline is sharpest near 40–45 mm, and a thin dark
  rod (top right) is in focus near 23 mm (at S = 195). Nothing else comes into focus, so it is of little use for
  geometry.
- **Outputs:**
  - `data/real_v2/{wrist_v2,set2}/{dts-fbp,os-sart}_{0.198mm,1mm}.tif` (with JSON sidecars);
  - `viewer.html` (1 mm slices, both methods side by side).
  - Figures: `recon_wrist_v2.png`, `recon_set2.png`.

## 7. Old-software slices (sets 3 and 4): qualitative

No projections, so no reconstruction of ours to compare with and no metrics against them (as asked).
Figures: `old_dental_sheet.png`, `old_fish_sheet.png`, `old_{dental,fish}_lines.png`.

| | dental (set 3) | fish (set 4) |
|---|---|---|
| slices, size | 32, 1000 × 1200 (portrait) | 20, 512 × 512 |
| in focus | crowns ~slices 8–10, roots 12–16; jaw outline drifts through 16–24 | head ~5–9, vertebrae and ribs 9–12, bright blob (otolith or metal?) 17–19 |
| replicated straight lines | vertical, at the right edge, 6–7 copies | horizontal band near the top, 7 copies |
| line spacing per slice [M] | 1.61 px (rms 0.12 over 25 slices) | 0.40 px (rms 0.04 over 9 slices) |
| spacing = 0 at slice [M] | 0.94 | −0.3 |
| fraction of pixels at 0 [M] | 18 % (slice 0) → 30 % (slices 15–20), then 28 % | 3 % → 33 %, monotonic |

- **Artifacts** [M]:
  - Both stacks show the same mechanism v1 found in the Fingers AVI: a straight detector-fixed feature (detector edge or
    a defect line), backprojected once per emitter row or column, appears as 7 parallel copies whose spacing grows with
    height.
  - Dental also has the following:
    - a fan of diagonal streaks from the lower-right corner in the lowest slices;
    - horizontal banding along the bottom edge;
    - a cross-hatched texture in the upper slices (27–31), at the emitter-grid period;
    - vertical streaks across the right third.
  - Fish has the top band, a vertical line at the right that splits with depth, and a rising mean brightness toward the
    top slices.
  - Both clip negative values to 0, and the zero region grows with slice index: the field seen by all emitters shrinks
    with height [I].
- **Slice spacing and count** [M + I]:
  - Copies from adjacent emitters are pitch · z / S apart at height z. So the spacing per slice is
    10 mm · Δz / (S · v), where v is the slice pixel size.
  - The measured slopes give Δz/S = 0.161·v (dental) and 0.040·v (fish), v in mm. The zero crossing puts the detector
    plane at slice ≈ 1 (dental) and ≈ −0.3 (fish).
  - With unknown v and S, Δz is not determined. Examples [A]:
    - dental with v = 0.099 mm (the 1152 grid of cmd.txt) and S = 120: Δz = 1.9 mm, 32 slices ≈ 61 mm, close to
      cmd.txt's 64 mm object height;
    - with v = 0.0495 mm: 1 mm, 32 mm.
  - The count of copies (7) matches the 7 emitter columns or rows of the keys grids.
- **Our stacks** [M, by eye; not quantified]:
  - No interior replicated lines like the dental and fish ones. Flat-field and defect handling remove the
    detector-fixed structure that the old software replicated.
  - They do show stepped copies of the detector edge at the volume border, mostly far from the object range (e.g.
    set 2 at z = 64 mm). v1 measured the same at the ends of its range.

## 8. Questions for Gil and answers (2026-10-27)

| | Question | Answer | Consequence |
|---|---|---|---|
| 1 | S per setup; detector cover-to-sensor distance | not known | S stays inferred: A ≈ 145 mm (air, model), B ≈ 195 mm (ring), both PLACEHOLDER; the setup-B tension (§5.3) stays open |
| 2 | Emitter pitch | **10 mm** | units reading ruled out; air peaks are displaced from the sources (§4) |
| 3 | Native air images for the new wrist | none | the wrist keeps the borrowed, 2× upsampled 1152 flats (3.5 % pixel-level noise) |
| 4 | Projections, detector, slice pixel size for dental and fish | none | §7 stays qualitative; their slice spacing cannot be converted to mm |
| 5 | Set 2: object, raw data | not known | PNG scale assumed (s = 12619.5), S assumed (195 mm); set 2 is not useful for geometry |
| 6 | Andy040618's own keys.txt | none | setup A's grid offset only from the air fit (assumes the aimed model) |
| 7 | Keys units and origin | mm (pitch answer); origin not known | the keys origin at the detector corner stays an assumption; it only shears the volume |

**What would still settle S** (if this system is ever imaged again):
- a known-height object (two-plane bead phantom, or a ruler and a coin on spacers);
- a tape measure from emitter face to detector cover, plus the detector's datasheet cover-to-sensor offset.

## 9. What changed

- **Code:**
  - `read_keys`, `keys_rc`;
  - the geometry-free parallax tools in `selfcal`;
  - the pitch-fixed, aimed-emission air model in `air`;
  - `scripts/real_v2.py`;
  - three tests.
  The v1 code and results are unchanged.
- **Configs:** one per setup:
  - `legacy_2018_setup_a.yaml` (Andy040618);
  - `legacy_2018_setup_b.yaml` (1152 air, Andy/wrist, Fingers, new wrist);
  - `legacy_2018_set2.yaml`;
  - `legacy_2018_dental_slices.yaml`;
  - `legacy_2018_fish_slices.yaml`.
  Every inferred or assumed number in them is marked PLACEHOLDER. `legacy_emitter_2018.yaml` stays as v1's run
  config; its Fingers entry is now known to use the wrong flats (noted there).
- **docs/decisions.md** item 29 and `docs/STATUS.md` updated.
