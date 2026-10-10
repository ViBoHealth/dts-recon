# Raw scan format (mock, v1)

**PLACEHOLDER.** This is our own format for mock data. The real format still has to come from the hardware
team (brief §10, items 1–3). It records everything §10 asks the hardware to provide, so we can hand it to them
as a strawman. Reader: `src/dtsrecon/io/scan.py`. When the real format arrives, only a reader that builds
the same `Scan` object needs to change; preprocessing and reconstruction stay as they are.

## Layout

```
my_scan/
  scan.yaml            manifest (below)
  frames/
    dark_000.raw ...   detector frames, one file each, no header
    flat_000.raw ...
    proj_000_0.raw ... proj_<position>_<repeat>.raw
  truth.json           mock scans only: the true source positions and errors (not used by the pipeline)
```

## Frames

- **Layout:** headerless, row-major `[row, col]`. The column index runs along +x (`det_u`, the detector's 30 cm
  side) and the row index along +y (`det_v`). This follows the world frame in brief §4.1.
- **Pixel type:** `dtype` and `byte_order` come from the manifest. The default is little-endian `uint16`.
  `bit_depth` sets the saturation level (2^bits − 1).
- **Size check:** a file's size must equal `n_rows × n_cols × bytes per pixel`.

## Manifest (`scan.yaml`)

| Field | Meaning |
|---|---|
| `format`, `version` | `dts-raw`, `1` |
| `placeholder` | `true` for mock or unverified data |
| `description` | free text |
| `detector` | `n_cols`, `n_rows`, `pixel_pitch_u`, `pixel_pitch_v` (mm), `dtype`, `byte_order`, `bit_depth` |
| `geometry.sid_mm` | nominal source-plane height above the detector face (mm) |
| `geometry.calibration` | optional explicit geometry YAML (from `dts calibrate`), one entry per source position, path relative to the scan |
| `object_z_range_mm` | optional `[z_min, z_max]`, used as the default reconstruction range |
| `frames` | list of frames (below) |

Each frame has:

| Field | Frames | Meaning |
|---|---|---|
| `file` | all | path relative to the scan directory |
| `kind` | all | `dark` (no beam), `flat` (beam, no object) or `projection` |
| `timestamp_s` | all, optional | acquisition time |
| `exposure` | flat, projection | `kvp`, `ua` (µA), `ms`. If both flats and a projection give µA and ms, the flat is scaled by the charge ratio |
| `position_index` | projection | source position, 0 … N−1 with no gaps |
| `exposure_index` | projection | repeat number at that position. Repeats are averaged |
| `encoder_x_mm`, `encoder_y_mm` | projection | encoder reading of the source position in the detector frame |

## Geometry

The scan's geometry comes from one of two places:

1. **Without a calibration file:** source *i* is placed at (mean encoder x, mean encoder y, `sid_mm`), and the detector is the
   world frame.
2. **With a calibration file:** the explicit per-position vectors in the calibration file are used instead.

Encoders do not see systematic stage errors (offset, SID, scale, rotation, tilt). M5 measured what those
errors do to the image: `docs/decisions.md` item 24.

## Open questions for the hardware team

These cover brief §10:
- the file type, bit depth and byte order of the real detector;
- whether there is a header;
- how darks and flats are acquired, and how often;
- the frame of the encoder axes;
- how many exposures are taken per position;
- whether there is per-frame exposure metadata;
- whether they will deliver DICOM or raw files plus metadata.
