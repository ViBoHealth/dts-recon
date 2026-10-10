"""Slice-stack export: float32 TIFF (ImageJ hyperstack, mm spacing) + JSON sidecar, and DICOM.

Array layout everywhere: [z, y, x], slice k at height z_k above the detector face; row index increases with
+y, column index with +x (brief §4.1). Viewers draw row 0 at the top, so +y points down on screen.
Values: reconstructed linear attenuation, 1/mm (DTS values are not quantitative; brief §7).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from ..geometry.schema import VolumeSpec


def _vol_dict(vol: VolumeSpec) -> dict:
    return {"nx": vol.nx, "ny": vol.ny, "nz": vol.nz, "voxel_xy_mm": vol.voxel_xy, "dz_mm": vol.dz, "center_mm": list(vol.center),
            "x0_mm": float(vol.axis("x")[0]), "y0_mm": float(vol.axis("y")[0]), "z_mm": [float(z) for z in vol.axis("z")]}


def write_tiff(path: str | Path, f: np.ndarray, vol: VolumeSpec, meta: dict | None = None) -> Path:
    """ImageJ-compatible float32 stack (Fiji shows mm spacing) plus <path>.json with the full grid and meta.
    Stacks over ~4 GB are written as BigTIFF (classic TIFF offsets are 32-bit; Fiji opens BigTIFF via
    Bio-Formats); the grid is then only in the sidecar and the resolution tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(f, dtype=np.float32)
    res = (1 / vol.voxel_xy, 1 / vol.voxel_xy)
    if a.nbytes < 3.9 * 2**30:
        tifffile.imwrite(path, a, imagej=True, resolution=res, metadata={"spacing": vol.dz, "unit": "mm", "axes": "ZYX"})
    else:
        tifffile.imwrite(path, a, bigtiff=True, resolution=res, resolutionunit="NONE",
                         metadata={"spacing": vol.dz, "unit": "mm", "axes": "ZYX"})
    side = {"volume": _vol_dict(vol), "units": "1/mm (not quantitative)",
            "layout": "[z, y, x]; row index +y, column index +x; z above the detector face", **(meta or {})}
    path.with_suffix(".json").write_text(json.dumps(side, indent=2, default=str))
    return path


def read_tiff(path: str | Path) -> tuple[np.ndarray, VolumeSpec, dict]:
    """Stack, grid (from the sidecar if present, else from the ImageJ tags, centered at z of slice 0 = dz) and
    the sidecar dict."""
    path = Path(path)
    f = tifffile.imread(path)
    side = path.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
        v = meta["volume"]
        return f, VolumeSpec(v["nx"], v["ny"], v["nz"], v["voxel_xy_mm"], v["dz_mm"], tuple(v["center_mm"])), meta
    with tifffile.TiffFile(path) as tf:
        ij = tf.imagej_metadata or {}
        xres = tf.pages[0].tags["XResolution"].value
    vxy = xres[1] / xres[0]
    dz = float(ij.get("spacing", vxy))
    nz, ny, nx = f.shape
    return f, VolumeSpec.from_z_range(nx, ny, nz, vxy, dz, dz), {}


def write_dicom(out_dir: str | Path, f: np.ndarray, vol: VolumeSpec, series_description: str = "DTS reconstruction",
                patient_name: str = "PHANTOM^SIMULATED", patient_id: str = "SIM0001", meta: dict | None = None) -> list[Path]:
    """One file per slice (CT Image Storage, 16-bit with rescale slope), patient coordinates = detector frame
    in mm (ImagePositionPatient = (x0, y0, z_k), orientation rows +x, columns +y). Research output only:
    SeriesDescription and ImageComments say so. Needs pydicom (optional extra `dicom`)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(np.abs(f))) or 1.0
    slope = vmax / 32000.0
    study, series, frame_uid = generate_uid(), generate_uid(), generate_uid()
    now = datetime.now()
    x0, y0 = float(vol.axis("x")[0]), float(vol.axis("y")[0])
    paths = []
    comment = "Research use only, not for diagnosis. Simulated/prototype DTS. " + json.dumps(meta or {}, default=str)[:900]
    for k, z in enumerate(vol.axis("z")):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, fm.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame_uid
        ds.Modality = "CT"
        ds.ImageType = ["DERIVED", "SECONDARY", "TOMOSYNTHESIS"]
        ds.PatientName, ds.PatientID = patient_name, patient_id
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
        ds.Manufacturer = "dts-recon"
        ds.SeriesDescription = f"{series_description} (research)"
        ds.ImageComments = comment
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.ImagePositionPatient = [x0, y0, float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [vol.voxel_xy, vol.voxel_xy]
        ds.SliceThickness = ds.SpacingBetweenSlices = vol.dz
        ds.SliceLocation = float(z)
        ds.Rows, ds.Columns = vol.ny, vol.nx
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = f"{slope:.6g}", 0, "US"
        ds.PixelData = np.round(f[k] / float(f"{slope:.6g}")).clip(-32768, 32767).astype("<i2").tobytes()
        p = out / f"slice_{k:04d}.dcm"
        pydicom.dcmwrite(p, ds, enforce_file_format=True)
        paths.append(p)
    return paths
