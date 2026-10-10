# Using the `dts` tool

This page takes you from raw detector frames to a slice stack you can scroll through. It uses mock data because
no real data exists yet. Setup is in the README (Python venv, LEAP CPU build).

## 1. Get a scan

A scan is a directory with a `scan.yaml` and the raw frames. The format is described in `docs/raw_format.md`.
To make mock scans:

```bash
.venv/bin/dts mock-scan data/cal_scan --phantom calibration    # bead phantom, full 0.1 mm panel (~620 MB)
.venv/bin/dts mock-scan data/dog_scan --phantom dental         # medium-dog dental phantom, 0.4 mm panel
.venv/bin/dts info data/dog_scan
```

- Mock scans use the dental geometry (`configs/geometry_dental.yaml`: 6 × 6 sources over 400 mm, SID 500).
  Change it with `--config`.
- By default they include example stage errors that the encoders do not see. `--no-stage-errors` turns these off.
- By default they use realistic physics (polyenergetic, scatter, focal spot, lag). The detector model adds
  noise, offset and gain maps, and dead or hot pixels. `--ideal` gives monoenergetic physics with no scatter.
- All values are placeholders.

## 2. Calibrate the geometry (once per stage set-up)

```bash
.venv/bin/dts calibrate data/cal_scan -o data/stage_geometry.yaml
```

- This finds every source position from the bead shadows (M5).
- It prints the fit residual, the estimated uncertainty, and how far the sources moved from the encoder
  positions. A JSON report is written next to the YAML.
- It needs full-resolution frames (≤ 0.2 mm pixels).

## 3. Reconstruct

```bash
.venv/bin/dts recon data/dog_scan -o data/recon --method os-sart --calibration data/stage_geometry.yaml
```

| Option | Default | |
|---|---|---|
| `--method` | `os-sart` | `asd-pocs` (sharper in depth, slower, ~10x the memory); `dts-fbp` (fastest, weakest on wide low-contrast objects) |
| `--calibration` | the scan's, else encoders | geometry YAML from step 2 |
| `--binning` | to ~0.4 mm pixels | detector binning before reconstruction |
| `--voxel` | binned pixel size | cubic voxels (LEAP) |
| `--z` | object range in the scan ± 5 mm, else `10,210` | e.g. `--z 10,120` |
| `--margin` | 20 mm | volume extends past the detector in x/y. Iterative methods need the whole object inside the volume |
| `--iterations` | 10 | |
| `--slice-mm` | none | average slices to this thickness after reconstruction |
| `--dicom` | off | also write a DICOM series (needs `pydicom`: `-e '.[dicom]'`) |

Outputs:
- `<method>.tif`: a float32 stack `[z, y, x]` with mm spacing, which opens in Fiji/ImageJ.
- `<method>.json`: the grid, slice heights, preprocessing summary and timings.
- `<method>_dicom/` if you asked for `--dicom`.

## 4. Look at it

```bash
.venv/bin/dts view data/recon/os-sart.tif                       # napari; z slider at the bottom
.venv/bin/dts view data/recon/os-sart.tif data/recon/asd-pocs.tif   # side by side
```

- napari is the optional extra `viz` (`uv pip install --python .venv/bin/python -e '.[viz]'`).
- The TIFF also opens in Fiji. The DICOM series opens in any DICOM viewer, such as Horos or 3D Slicer.
- Row 0 is at the top, so +y points down on screen.

## Whole chain in one go

`scripts/m6_end_to_end.sh` runs all of the above. It writes to `data/m6/` and logs timings and peak memory to
`reports/m6/e2e_run.log`.
