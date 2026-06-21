"""
4DCT VOI batch-thresholding tool  —  extended version
======================================================

Extends the original MIP-only tool to process ALL series types:
  - MIP series  (Series Number 100, or 'MIP' in Image Type + Series Description)
  - Phase series (T=XX% detected in Series Description)
  - raw_4DCT     (plain 'Thorax 4DCT' or similar with no phase/MIP label)

Workflow:
  1. Pick a parent folder containing study/series subfolders.
  2. Tool walks each subfolder, classifies it as MIP / phase / raw_4DCT / unknown.
  3. User defines a VOI (centre + size in patient mm) and HU threshold range
     on the first MIP found, with 3-plane preview and live mask overlay.
  4. Config is saved to JSON; same VOI+threshold is applied to all series.
  5. Results written to:
       - CSV  (one row per series)
       - SQLite database  (same data, queryable)

Database schema (table: measurements):
  exam_id, series_folder, series_type (MIP/T=0%/T=10% etc),
  phase_pct (numeric, NULL for MIP/raw), patient_id, study_instance_uid,
  series_instance_uid, series_number, series_description,
  acquisition_date, station_name, manufacturer_model,
  voi_centre_x/y/z_mm, voi_size_x/y/z_mm, threshold_hu_min/max,
  n_voxels_in_band, volume_cc, centroid_x/y/z_mm, notes

Requirements:
  pip install pydicom numpy matplotlib pandas
  (sqlite3 is part of the Python standard library)

Coordinate conventions:
  Patient LPS millimetres (DICOM standard).
  3D HU array indexed [k, i, j] = [slice, row, column], k=0 at lowest z.
"""

import os
import re
import json
import sqlite3
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pydicom

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd


# --------------------------------------------------------------------------
# Series classification
# --------------------------------------------------------------------------

def classify_series(ds) -> Tuple[str, Optional[int]]:
    """
    Classify a DICOM series by type.

    Returns (series_type, phase_pct):
      ('MIP',   None)   — MIP series
      ('T=XX%', XX)     — phase series, phase_pct is the integer percentage
      ('raw_4DCT', None)— plain 4DCT with no phase/MIP info
      ('unknown', None) — unrecognised
    """
    series_desc  = str(getattr(ds, "SeriesDescription", "") or "").strip()
    series_num   = str(getattr(ds, "SeriesNumber",      "") or "").strip()
    image_type   = getattr(ds, "ImageType", None)

    # MIP: series number 100, or 'MIP' in both Image Type and Series Description
    if series_num == "100":
        return "MIP", None
    if image_type is not None:
        type_str = "\\".join(str(x) for x in image_type).upper()
        if "MIP" in type_str and "MIP" in series_desc.upper():
            return "MIP", None
    if "mip" in series_desc.lower():
        return "MIP", None

    # Phase: T=XX% in Series Description
    m = re.search(r"T=(\d+)%", series_desc, re.IGNORECASE)
    if m:
        pct = int(m.group(1))
        return f"T={pct}%", pct

    # Raw 4DCT: plain series description with no phase info
    if "4dct" in series_desc.lower() or "thorax" in series_desc.lower():
        return "raw_4DCT", None

    return "unknown", None


# --------------------------------------------------------------------------
# DICOM IO — walk folder tree
# --------------------------------------------------------------------------

_ARIA_LAYER_RE = re.compile(r"^(SE|ST|PA)\d+$", re.IGNORECASE)


def read_first_dicom(folder: str):
    try:
        names = sorted(os.listdir(folder))
    except Exception:
        return None
    for name in names:
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        try:
            return pydicom.dcmread(full, stop_before_pixels=True)
        except Exception:
            continue
    return None


def _infer_study_path(series_folder: str, root: str) -> str:
    p    = os.path.normpath(series_folder)
    root = os.path.normpath(root)
    while True:
        if p == root:
            return p
        head, tail = os.path.split(p)
        if not _ARIA_LAYER_RE.match(tail):
            return p
        if head == p:
            return p
        p = head


def walk_parent_folder(parent: str) -> List[Dict]:
    """
    Recursively walk *parent* and return one record per classifiable series folder.
    Includes MIP, phase, raw_4DCT, and unknown series (skips folders with no DICOMs).
    """
    parent  = os.path.normpath(parent)
    records = []

    def consider(dirpath: str):
        ds = read_first_dicom(dirpath)
        if ds is None:
            return
        series_type, phase_pct = classify_series(ds)
        n_files = sum(1 for f in os.listdir(dirpath)
                      if os.path.isfile(os.path.join(dirpath, f)))

        # Extract exam ID from StudyID tag
        # Use the immediate parent folder name as exam_id
        # (e.g. "16929 3") so it matches the user's own organisation
        exam_id = os.path.basename(_infer_study_path(dirpath, parent))

        records.append({
            "study_path":   _infer_study_path(dirpath, parent),
            "series_folder": dirpath,
            "series_type":  series_type,
            "phase_pct":    phase_pct,
            "exam_id":      exam_id,
            "first_ds":     ds,
            "n_files":      n_files,
        })

    consider(parent)
    for dirpath, dirnames, _ in os.walk(parent):
        dirnames.sort()
        for d in list(dirnames):
            consider(os.path.join(dirpath, d))

    # de-duplicate
    seen, unique = set(), []
    for r in records:
        sf = r["series_folder"]
        if sf not in seen:
            seen.add(sf)
            unique.append(r)
    return unique


# --------------------------------------------------------------------------
# Volume loading and geometry
# --------------------------------------------------------------------------

@dataclass
class VolumeGeometry:
    origin_xyz_mm:       Tuple[float, float, float]
    col_spacing_mm:      float
    row_spacing_mm:      float
    slice_positions_z_mm: np.ndarray
    n_rows:  int
    n_cols:  int
    n_slices: int

    @property
    def slice_spacing_mm(self) -> float:
        if self.n_slices < 2:
            return 1.0
        return float(np.median(np.abs(np.diff(self.slice_positions_z_mm))))

    @property
    def voxel_volume_cc(self) -> float:
        return (self.col_spacing_mm * self.row_spacing_mm
                * self.slice_spacing_mm) / 1000.0


@dataclass
class LoadedVolume:
    hu:          np.ndarray
    geom:        VolumeGeometry
    series_meta: Dict


def load_volume(folder: str) -> Optional[LoadedVolume]:
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None

    dsets = []
    for path in files:
        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "pixel_array"):
                dsets.append(ds)
        except Exception:
            continue

    if not dsets:
        return None

    def z_key(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0))

    dsets.sort(key=z_key)
    ref  = dsets[0]
    rows = int(ref.Rows)
    cols = int(ref.Columns)
    n    = len(dsets)

    ipp0 = [float(v) for v in ref.ImagePositionPatient]
    ps   = ref.PixelSpacing
    z_positions = np.array([float(d.ImagePositionPatient[2]) for d in dsets],
                           dtype=np.float64)

    hu = np.zeros((n, rows, cols), dtype=np.float32)
    for k, ds in enumerate(dsets):
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu[k]     = ds.pixel_array.astype(np.float32) * slope + intercept

    geom = VolumeGeometry(
        origin_xyz_mm        = (ipp0[0], ipp0[1], z_positions[0]),
        col_spacing_mm       = float(ps[1]),
        row_spacing_mm       = float(ps[0]),
        slice_positions_z_mm = z_positions,
        n_rows   = rows,
        n_cols   = cols,
        n_slices = n,
    )

    meta = {
        "PatientID":            str(getattr(ref, "PatientID",            "") or ""),
        "StudyID":              str(getattr(ref, "StudyID",              "") or ""),
        "StudyInstanceUID":     str(getattr(ref, "StudyInstanceUID",     "") or ""),
        "SeriesInstanceUID":    str(getattr(ref, "SeriesInstanceUID",    "") or ""),
        "SeriesNumber":         str(getattr(ref, "SeriesNumber",         "") or ""),
        "SeriesDescription":    str(getattr(ref, "SeriesDescription",    "") or ""),
        "AcquisitionDate":      str(getattr(ref, "AcquisitionDate",      "") or ""),
        "AcquisitionTime":      str(getattr(ref, "AcquisitionTime",      "") or ""),
        "StationName":          str(getattr(ref, "StationName",          "") or ""),
        "ManufacturerModelName":str(getattr(ref, "ManufacturerModelName","") or ""),
        "InstitutionName":      str(getattr(ref, "InstitutionName",      "") or ""),
    }

    return LoadedVolume(hu=hu, geom=geom, series_meta=meta)


# --------------------------------------------------------------------------
# Coordinate helpers
# --------------------------------------------------------------------------

def voxel_to_mm(geom, k, i, j):
    x = geom.origin_xyz_mm[0] + j * geom.col_spacing_mm
    y = geom.origin_xyz_mm[1] + i * geom.row_spacing_mm
    if geom.n_slices == 1:
        z = geom.slice_positions_z_mm[0]
    else:
        kf = float(np.clip(k, 0, geom.n_slices - 1))
        k0 = int(np.floor(kf));  k1 = min(k0 + 1, geom.n_slices - 1)
        t  = kf - k0
        z  = (1 - t) * geom.slice_positions_z_mm[k0] + t * geom.slice_positions_z_mm[k1]
    return float(x), float(y), float(z)


def mm_to_voxel(geom, x_mm, y_mm, z_mm):
    j = (x_mm - geom.origin_xyz_mm[0]) / geom.col_spacing_mm
    i = (y_mm - geom.origin_xyz_mm[1]) / geom.row_spacing_mm
    z_arr = geom.slice_positions_z_mm
    if z_mm <= z_arr[0]:
        k = 0.0
    elif z_mm >= z_arr[-1]:
        k = float(len(z_arr) - 1)
    else:
        idx = max(0, min(int(np.searchsorted(z_arr, z_mm)) - 1, len(z_arr) - 2))
        z0, z1 = z_arr[idx], z_arr[idx + 1]
        k = float(idx) + (0.0 if z1 == z0 else (z_mm - z0) / (z1 - z0))
    return float(k), float(i), float(j)


# --------------------------------------------------------------------------
# Configuration (VOI + threshold)
# --------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    voi_centre_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    voi_size_mm:   Tuple[float, float, float] = (40.0, 40.0, 40.0)
    threshold_hu:  Tuple[float, float]        = (-100.0, 100.0)

    def to_json(self) -> str:
        return json.dumps({
            "voi_centre_mm": list(self.voi_centre_mm),
            "voi_size_mm":   list(self.voi_size_mm),
            "threshold_hu":  list(self.threshold_hu),
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AnalysisConfig":
        d = json.loads(s)
        return cls(
            voi_centre_mm = tuple(d["voi_centre_mm"]),
            voi_size_mm   = tuple(d["voi_size_mm"]),
            threshold_hu  = tuple(d["threshold_hu"]),
        )


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    n_voxels_in_band: int
    volume_cc:        float
    centroid_mm:      Tuple[float, float, float]
    voi_voxel_bounds: Dict
    notes: str = ""


def voi_voxel_bounds(geom, cfg):
    cx, cy, cz = cfg.voi_centre_mm
    sx, sy, sz = cfg.voi_size_mm
    k_lo, i_lo, j_lo = mm_to_voxel(geom, cx - sx/2, cy - sy/2, cz - sz/2)
    k_hi, i_hi, j_hi = mm_to_voxel(geom, cx + sx/2, cy + sy/2, cz + sz/2)

    def clip(lo, hi, maxval):
        lo_i = max(0, int(np.floor(min(lo, hi))))
        hi_i = min(maxval, int(np.ceil(max(lo, hi))) + 1)
        return lo_i, hi_i

    return {
        "k": clip(k_lo, k_hi, geom.n_slices),
        "i": clip(i_lo, i_hi, geom.n_rows),
        "j": clip(j_lo, j_hi, geom.n_cols),
    }


def analyse(volume: LoadedVolume, cfg: AnalysisConfig) -> AnalysisResult:
    b  = voi_voxel_bounds(volume.geom, cfg)
    k0, k1 = b["k"];  i0, i1 = b["i"];  j0, j1 = b["j"]
    notes  = []

    if k1 <= k0 or i1 <= i0 or j1 <= j0:
        return AnalysisResult(0, 0.0, (float("nan"),)*3, b, "VOI fully outside volume")

    sub  = volume.hu[k0:k1, i0:i1, j0:j1]
    mask = (sub >= cfg.threshold_hu[0]) & (sub <= cfg.threshold_hu[1])
    n_in = int(mask.sum())

    if n_in == 0:
        return AnalysisResult(0, 0.0, (float("nan"),)*3, b, "no voxels in HU band within VOI")

    vol_cc = n_in * volume.geom.voxel_volume_cc
    ks, is_, js = np.where(mask)
    cx, cy, cz  = voxel_to_mm(volume.geom,
                               ks.mean() + k0,
                               is_.mean() + i0,
                               js.mean() + j0)
    return AnalysisResult(n_in, float(vol_cc), (cx, cy, cz), b, "; ".join(notes))


# --------------------------------------------------------------------------
# QC image output
# --------------------------------------------------------------------------

def save_qc_image(volume: LoadedVolume, cfg: AnalysisConfig,
                  result: AnalysisResult, rec: Dict, output_dir: str):
    """
    Save a 3-panel QC image (axial, coronal, sagittal) showing the CT with
    the HU mask overlaid in red, centred on the VOI centroid.
    One image saved per series for visual review.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for batch saving
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    geom = volume.geom
    b    = result.voi_voxel_bounds

    # Use centroid slice if valid, otherwise use VOI centre
    cx, cy, cz = result.centroid_mm
    if any(np.isnan(v) for v in (cx, cy, cz)):
        cx, cy, cz = cfg.voi_centre_mm

    kc, ic, jc = mm_to_voxel(geom, cx, cy, cz)
    kc = int(np.clip(round(kc), 0, geom.n_slices - 1))
    ic = int(np.clip(round(ic), 0, geom.n_rows   - 1))
    jc = int(np.clip(round(jc), 0, geom.n_cols   - 1))

    # Build full mask within VOI bounds
    full_mask = np.zeros_like(volume.hu, dtype=bool)
    k0, k1 = b["k"];  i0, i1 = b["i"];  j0, j1 = b["j"]
    if k1 > k0 and i1 > i0 and j1 > j0:
        sub = volume.hu[k0:k1, i0:i1, j0:j1]
        sub_mask = (sub >= cfg.threshold_hu[0]) & (sub <= cfg.threshold_hu[1])
        full_mask[k0:k1, i0:i1, j0:j1] = sub_mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = -200.0, 300.0

    slices = [
        (axes[0], volume.hu[kc, :, :],    full_mask[kc, :, :],
         f"Axial  z={geom.slice_positions_z_mm[kc]:.1f} mm", "upper"),
        (axes[1], volume.hu[:, ic, :],    full_mask[:, ic, :],
         f"Coronal  y slice {ic}",  "lower"),
        (axes[2], volume.hu[:, :, jc],    full_mask[:, :, jc],
         f"Sagittal  x slice {jc}", "lower"),
    ]

    for ax, img, mask, title, origin in slices:
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin=origin, aspect="auto")
        ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask, dtype=float)),
                  cmap="autumn", alpha=0.5, origin=origin, aspect="auto",
                  vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Title with key info
    exam_id     = rec.get("exam_id", "")
    series_type = rec.get("series_type", "")
    vol_cc      = result.volume_cc
    notes       = result.notes
    cx, cy, cz  = result.centroid_mm

    # Format centroid — show as (x, y, z) mm or N/A if not computed
    if any(np.isnan(v) for v in (cx, cy, cz)):
        centroid_str = "N/A"
    else:
        centroid_str = f"({cx:.1f}, {cy:.1f}, {cz:.1f}) mm"

    suptitle = (
        f"Exam: {exam_id}  |  Phase: {series_type}  |  "
        f"Volume: {vol_cc:.4f} cc  |  Centroid: {centroid_str}"
    )
    if notes:
        suptitle += f"  |  ⚠ {notes}"
    fig.suptitle(suptitle, fontsize=9, fontweight="bold")

    # Red patch legend
    patch = mpatches.Patch(color="yellow", alpha=0.7, label=f"Contoured volume  HU {cfg.threshold_hu[0]}–{cfg.threshold_hu[1]}")
    fig.legend(handles=[patch], loc="lower center", fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Build filename
    safe_exam   = "".join(c if c.isalnum() or c in " _-" else "_" for c in exam_id)
    safe_series = "".join(c if c.isalnum() or c in "=%" else "_" for c in series_type)
    filename    = f"QC_{safe_exam}_{safe_series}.png".replace(" ", "_")
    filepath    = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filepath


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

DB_COLUMNS = [
    "exam_id", "series_folder", "series_type", "phase_pct",
    "patient_id", "study_instance_uid", "series_instance_uid",
    "series_number", "series_description",
    "acquisition_date", "station_name", "manufacturer_model",
    "voi_centre_x_mm", "voi_centre_y_mm", "voi_centre_z_mm",
    "voi_size_x_mm",   "voi_size_y_mm",   "voi_size_z_mm",
    "threshold_hu_min", "threshold_hu_max",
    "n_voxels_in_band", "volume_cc",
    "centroid_x_mm", "centroid_y_mm", "centroid_z_mm",
    "notes",
]

CSV_COLUMNS = DB_COLUMNS   # same columns for CSV export


def result_to_row(rec: Dict, meta: Dict, cfg: AnalysisConfig,
                  result: AnalysisResult) -> Dict:
    return {
        "exam_id":              rec.get("exam_id", ""),
        "series_folder":        rec["series_folder"],
        "series_type":          rec["series_type"],
        "phase_pct":            rec["phase_pct"],       # int or None
        "patient_id":           meta.get("PatientID", ""),
        "study_instance_uid":   meta.get("StudyInstanceUID", ""),
        "series_instance_uid":  meta.get("SeriesInstanceUID", ""),
        "series_number":        meta.get("SeriesNumber", ""),
        "series_description":   meta.get("SeriesDescription", ""),
        "acquisition_date":     meta.get("AcquisitionDate", ""),
        "station_name":         meta.get("StationName", ""),
        "manufacturer_model":   meta.get("ManufacturerModelName", ""),
        "voi_centre_x_mm":      cfg.voi_centre_mm[0],
        "voi_centre_y_mm":      cfg.voi_centre_mm[1],
        "voi_centre_z_mm":      cfg.voi_centre_mm[2],
        "voi_size_x_mm":        cfg.voi_size_mm[0],
        "voi_size_y_mm":        cfg.voi_size_mm[1],
        "voi_size_z_mm":        cfg.voi_size_mm[2],
        "threshold_hu_min":     cfg.threshold_hu[0],
        "threshold_hu_max":     cfg.threshold_hu[1],
        "n_voxels_in_band":     result.n_voxels_in_band,
        "volume_cc":            result.volume_cc,
        "centroid_x_mm":        result.centroid_mm[0],
        "centroid_y_mm":        result.centroid_mm[1],
        "centroid_z_mm":        result.centroid_mm[2],
        "notes":                result.notes,
    }


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the measurements table exists."""
    conn = sqlite3.connect(db_path)
    cols_sql = ",\n    ".join(
        f"{c} REAL" if c in ("phase_pct",
                              "voi_centre_x_mm","voi_centre_y_mm","voi_centre_z_mm",
                              "voi_size_x_mm","voi_size_y_mm","voi_size_z_mm",
                              "threshold_hu_min","threshold_hu_max",
                              "n_voxels_in_band","volume_cc",
                              "centroid_x_mm","centroid_y_mm","centroid_z_mm")
        else f"{c} TEXT"
        for c in DB_COLUMNS
    )
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols_sql}
        )
    """)
    conn.commit()
    return conn


def insert_row(conn: sqlite3.Connection, row: Dict):
    placeholders = ", ".join("?" for _ in DB_COLUMNS)
    cols         = ", ".join(DB_COLUMNS)
    values       = [row.get(c) for c in DB_COLUMNS]
    conn.execute(f"INSERT INTO measurements ({cols}) VALUES ({placeholders})", values)
    conn.commit()


# --------------------------------------------------------------------------
# VOI placement window (3-plane interactive)
# --------------------------------------------------------------------------

class VoiSetupWindow:
    def __init__(self, master, volume: LoadedVolume, cfg: AnalysisConfig):
        self.master = master
        self.vol    = volume
        self.cfg    = AnalysisConfig(
            voi_centre_mm = cfg.voi_centre_mm,
            voi_size_mm   = cfg.voi_size_mm,
            threshold_hu  = cfg.threshold_hu,
        )
        if cfg.voi_centre_mm == (0.0, 0.0, 0.0):
            kc = self.vol.geom.n_slices // 2
            ic = self.vol.geom.n_rows   // 2
            jc = self.vol.geom.n_cols   // 2
            self.cfg.voi_centre_mm = voxel_to_mm(self.vol.geom, kc, ic, jc)

        self.committed = False
        self.top = tk.Toplevel(master)
        self.top.title("VOI placement and threshold")
        self.top.geometry("1200x850")

        main = ttk.Frame(self.top)
        main.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        self.ax_axial   = axes[0, 0]
        self.ax_coronal = axes[0, 1]
        self.ax_sagittal= axes[1, 0]
        axes[1, 1].axis("off")
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl = ttk.Frame(main, padding=10)
        ctrl.pack(side="right", fill="y")
        self._build_controls(ctrl)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self._refresh_all()

    def _build_controls(self, parent):
        for section, pairs, attr in [
            ("VOI centre (mm, LPS)", [("x", "var_cx"), ("y", "var_cy"), ("z", "var_cz")],
             "voi_centre_mm"),
            ("VOI size (mm)",        [("dx","var_sx"), ("dy","var_sy"), ("dz","var_sz")],
             "voi_size_mm"),
            ("Threshold HU",         [("min","var_hu_lo"), ("max","var_hu_hi")],
             "threshold_hu"),
        ]:
            ttk.Label(parent, text=section, font=("", 10, "bold")).pack(anchor="w")
            for lbl, varname in pairs:
                vals = getattr(self.cfg, attr)
                idx  = ["x","dx","min"].index(lbl) if lbl in ["x","dx","min"] else \
                       ["y","dy"].index(lbl) if lbl in ["y","dy"] else 1
                val  = vals[["x","y","z","dx","dy","dz","min","max"].index(lbl) % len(vals)]
                var  = tk.DoubleVar(value=val)
                setattr(self, varname, var)
                row = ttk.Frame(parent);  row.pack(fill="x", pady=1)
                ttk.Label(row, text=f"{lbl}:", width=4).pack(side="left")
                e = ttk.Entry(row, textvariable=var, width=12);  e.pack(side="left")
                e.bind("<Return>",   lambda ev: self._on_field_changed())
                e.bind("<FocusOut>", lambda ev: self._on_field_changed())
            ttk.Separator(parent).pack(fill="x", pady=8)

        ttk.Label(parent, text="Live readout", font=("",10,"bold")).pack(anchor="w")
        self.lbl_voxels = ttk.Label(parent, text="voxels: -");        self.lbl_voxels.pack(anchor="w")
        self.lbl_volume = ttk.Label(parent, text="volume (cc): -");   self.lbl_volume.pack(anchor="w")
        self.lbl_cx     = ttk.Label(parent, text="centroid x: -");    self.lbl_cx.pack(anchor="w")
        self.lbl_cy     = ttk.Label(parent, text="centroid y: -");    self.lbl_cy.pack(anchor="w")
        self.lbl_cz     = ttk.Label(parent, text="centroid z: -");    self.lbl_cz.pack(anchor="w")

        ttk.Separator(parent).pack(fill="x", pady=8)
        ttk.Label(parent, text="Click any plane to set\nVOI centre.", foreground="#555").pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=12)
        btns = ttk.Frame(parent);  btns.pack(fill="x")
        ttk.Button(btns, text="OK",     command=self._on_ok).pack(side="left", padx=2)
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="left", padx=2)

    def _read_fields_to_cfg(self) -> bool:
        try:
            self.cfg.voi_centre_mm = (float(self.var_cx.get()),
                                      float(self.var_cy.get()),
                                      float(self.var_cz.get()))
            self.cfg.voi_size_mm   = (max(0.1, float(self.var_sx.get())),
                                      max(0.1, float(self.var_sy.get())),
                                      max(0.1, float(self.var_sz.get())))
            self.cfg.threshold_hu  = (float(self.var_hu_lo.get()),
                                      float(self.var_hu_hi.get()))
            return True
        except Exception:
            return False

    def _on_field_changed(self):
        if self._read_fields_to_cfg():
            self._refresh_all()

    def _push_cfg_to_fields(self):
        self.var_cx.set(round(self.cfg.voi_centre_mm[0], 3))
        self.var_cy.set(round(self.cfg.voi_centre_mm[1], 3))
        self.var_cz.set(round(self.cfg.voi_centre_mm[2], 3))

    def _refresh_all(self):
        self._draw_planes()
        self._update_readout()
        self.canvas.draw_idle()

    def _slice_indices(self):
        cx, cy, cz = self.cfg.voi_centre_mm
        kf, ifv, jf = mm_to_voxel(self.vol.geom, cx, cy, cz)
        k = int(np.clip(round(kf),  0, self.vol.geom.n_slices - 1))
        i = int(np.clip(round(ifv), 0, self.vol.geom.n_rows   - 1))
        j = int(np.clip(round(jf),  0, self.vol.geom.n_cols   - 1))
        return k, i, j

    def _draw_planes(self):
        k, i_row, j_col = self._slice_indices()
        hu_lo, hu_hi = self.cfg.threshold_hu
        full_mask = (self.vol.hu >= hu_lo) & (self.vol.hu <= hu_hi)
        b = voi_voxel_bounds(self.vol.geom, self.cfg)
        voi_only = np.zeros_like(full_mask)
        voi_only[b["k"][0]:b["k"][1], b["i"][0]:b["i"][1], b["j"][0]:b["j"][1]] = \
            full_mask[b["k"][0]:b["k"][1], b["i"][0]:b["i"][1], b["j"][0]:b["j"][1]]

        vmin, vmax = -200.0, 300.0
        for ax, img, mask, title in [
            (self.ax_axial,    self.vol.hu[k, :, :],       voi_only[k, :, :],
             f"Axial  k={k}  z={self.vol.geom.slice_positions_z_mm[k]:.1f} mm"),
            (self.ax_coronal,  self.vol.hu[:, i_row, :],   voi_only[:, i_row, :],
             f"Coronal  i={i_row}"),
            (self.ax_sagittal, self.vol.hu[:, :, j_col],   voi_only[:, :, j_col],
             f"Sagittal  j={j_col}"),
        ]:
            ax.clear()
            origin = "upper" if ax is self.ax_axial else "lower"
            ax.imshow(img,  cmap="gray",   vmin=vmin, vmax=vmax, origin=origin, aspect="equal" if ax is self.ax_axial else "auto")
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45, origin=origin, aspect="equal" if ax is self.ax_axial else "auto")
            ax.set_title(title);  ax.set_xticks([]);  ax.set_yticks([])

        # draw VOI box and crosshair on axial
        cx, cy, cz = self.cfg.voi_centre_mm
        sx, sy, sz = self.cfg.voi_size_mm
        g = self.vol.geom
        j0 = (cx - sx/2 - g.origin_xyz_mm[0]) / g.col_spacing_mm
        j1 = (cx + sx/2 - g.origin_xyz_mm[0]) / g.col_spacing_mm
        i0 = (cy - sy/2 - g.origin_xyz_mm[1]) / g.row_spacing_mm
        i1 = (cy + sy/2 - g.origin_xyz_mm[1]) / g.row_spacing_mm
        self.ax_axial.add_patch(Rectangle((j0, i0), j1-j0, i1-i0,
                                edgecolor="cyan", facecolor="none", linewidth=1.0))
        jc = (cx - g.origin_xyz_mm[0]) / g.col_spacing_mm
        ic = (cy - g.origin_xyz_mm[1]) / g.row_spacing_mm
        self.ax_axial.axvline(jc, color="yellow", linewidth=0.5, alpha=0.6)
        self.ax_axial.axhline(ic, color="yellow", linewidth=0.5, alpha=0.6)

    def _update_readout(self):
        try:
            r = analyse(self.vol, self.cfg)
        except Exception as e:
            self.lbl_voxels.config(text=f"voxels: error ({e})")
            return
        self.lbl_voxels.config(text=f"voxels: {r.n_voxels_in_band}")
        self.lbl_volume.config(text=f"volume (cc): {r.volume_cc:.4f}")
        cx, cy, cz = r.centroid_mm
        for lbl, v in [(self.lbl_cx, cx), (self.lbl_cy, cy), (self.lbl_cz, cz)]:
            lbl.config(text=("-" if np.isnan(v) else f"{v:.3f} mm"))

    def on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        cx, cy, cz = self.cfg.voi_centre_mm
        g = self.vol.geom
        if event.inaxes is self.ax_axial:
            self.cfg.voi_centre_mm = (g.origin_xyz_mm[0] + x * g.col_spacing_mm,
                                      g.origin_xyz_mm[1] + y * g.row_spacing_mm, cz)
        elif event.inaxes is self.ax_coronal:
            ki = int(np.clip(round(y), 0, g.n_slices - 1))
            self.cfg.voi_centre_mm = (g.origin_xyz_mm[0] + x * g.col_spacing_mm,
                                      cy, float(g.slice_positions_z_mm[ki]))
        elif event.inaxes is self.ax_sagittal:
            ki = int(np.clip(round(y), 0, g.n_slices - 1))
            self.cfg.voi_centre_mm = (cx, g.origin_xyz_mm[1] + x * g.row_spacing_mm,
                                      float(g.slice_positions_z_mm[ki]))
        self._push_cfg_to_fields()
        self._refresh_all()

    def _on_ok(self):
        if not self._read_fields_to_cfg():
            messagebox.showerror("Error", "Invalid numeric input.")
            return
        self.committed = True
        self.top.destroy()

    def _on_cancel(self):
        self.committed = False
        self.top.destroy()


# --------------------------------------------------------------------------
# Results window
# --------------------------------------------------------------------------

class ResultsWindow:
    def __init__(self, master, rows: List[Dict], default_save_dir: str):
        self.rows            = rows
        self.default_save_dir = default_save_dir

        self.top = tk.Toplevel(master)
        self.top.title("Batch results")
        self.top.geometry("1400x600")

        bar = ttk.Frame(self.top);  bar.pack(fill="x")
        ttk.Button(bar, text="Export CSV",
                   command=self._on_export_csv).pack(side="left", padx=4, pady=4)
        ttk.Button(bar, text="Export SQLite DB",
                   command=self._on_export_db).pack(side="left", padx=4, pady=4)
        ttk.Label(bar, text=f"{len(rows)} rows").pack(side="left", padx=8)

        tree_frame = ttk.Frame(self.top);  tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=CSV_COLUMNS, show="headings")
        for c in CSV_COLUMNS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, stretch=False)
        vs = ttk.Scrollbar(tree_frame, orient="vertical",   command=self.tree.yview)
        hs = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1);  tree_frame.columnconfigure(0, weight=1)

        for r in rows:
            self.tree.insert("", "end",
                             values=[self._fmt(r.get(c, "")) for c in CSV_COLUMNS])

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4f}"
        return "" if v is None else str(v)

    def _on_export_csv(self):
        path = filedialog.asksaveasfilename(
            initialdir=self.default_save_dir,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Save results CSV",
        )
        if not path:
            return
        df = pd.DataFrame(self.rows)
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df[CSV_COLUMNS].to_csv(path, index=False)
        messagebox.showinfo("Saved", f"CSV written to:\n{path}")

    def _on_export_db(self):
        path = filedialog.asksaveasfilename(
            initialdir=self.default_save_dir,
            defaultextension=".db",
            filetypes=[("SQLite database", "*.db")],
            title="Save SQLite database",
        )
        if not path:
            return
        try:
            conn = init_db(path)
            for row in self.rows:
                insert_row(conn, row)
            conn.close()
            messagebox.showinfo("Saved", f"Database written to:\n{path}\n\nTable: measurements\nRows: {len(self.rows)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write database:\n{e}")


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root          = root
        self.root.title("4DCT VOI batch tool")
        self.root.geometry("750x480")
        self.parent_folder: Optional[str]    = None
        self.records:       List[Dict]       = []
        self.config:        AnalysisConfig   = AnalysisConfig()

        bar = ttk.Frame(root, padding=8);  bar.pack(fill="x")
        ttk.Button(bar, text="1. Pick parent folder", command=self.on_pick_folder).pack(side="left", padx=2)
        ttk.Button(bar, text="2. Set VOI / threshold", command=self.on_setup_voi).pack(side="left", padx=2)
        ttk.Button(bar, text="Save config",            command=self.on_save_cfg).pack(side="left", padx=2)
        ttk.Button(bar, text="Load config",            command=self.on_load_cfg).pack(side="left", padx=2)
        ttk.Button(bar, text="3. Run batch",           command=self.on_run_batch).pack(side="left", padx=2)

        # Filter checkboxes
        fbar = ttk.LabelFrame(root, text="Series types to process", padding=6)
        fbar.pack(fill="x", padx=8, pady=2)
        self.var_mip     = tk.BooleanVar(value=True)
        self.var_phase   = tk.BooleanVar(value=True)
        self.var_raw     = tk.BooleanVar(value=False)
        self.var_unknown = tk.BooleanVar(value=False)
        ttk.Checkbutton(fbar, text="MIP",         variable=self.var_mip).pack(side="left", padx=6)
        ttk.Checkbutton(fbar, text="Phase (T=XX%)", variable=self.var_phase).pack(side="left", padx=6)
        ttk.Checkbutton(fbar, text="raw_4DCT",    variable=self.var_raw).pack(side="left", padx=6)
        ttk.Checkbutton(fbar, text="unknown",     variable=self.var_unknown).pack(side="left", padx=6)

        info = ttk.LabelFrame(root, text="Status", padding=8)
        info.pack(fill="both", expand=True, padx=8, pady=4)
        self.txt = tk.Text(info, height=20, wrap="word");  self.txt.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="ready.")
        ttk.Label(root, textvariable=self.status_var,
                  relief="sunken", anchor="w").pack(fill="x", side="bottom")

        self._log("4DCT VOI batch tool ready.")
        self._log("Processes MIP and phase series (T=0% – T=90%).")
        self._log("Pick a parent folder to begin.")

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.root.update_idletasks()

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def on_pick_folder(self):
        folder = filedialog.askdirectory(title="Pick parent folder of studies")
        if not folder:
            return
        self.parent_folder = folder
        self._log(f"\nParent folder: {folder}")
        self._set_status("Scanning for series...")
        try:
            self.records = walk_parent_folder(folder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan folder:\n{e}")
            traceback.print_exc()
            self._set_status("error.")
            return

        # Summary by type
        from collections import Counter
        counts = Counter(r["series_type"] for r in self.records)
        self._log(f"Found {len(self.records)} series total:")
        for stype, n in sorted(counts.items()):
            self._log(f"  {stype:<15} {n} series")
        for r in self.records:
            self._log(f"  [{r['series_type']:<10}]  ExamID={r['exam_id']}  {os.path.basename(r['series_folder'])}  (n={r['n_files']})")
        self._set_status(f"{len(self.records)} series found.")

    def on_setup_voi(self):
        # Prefer first MIP for VOI placement; fall back to first phase
        candidates = [r for r in self.records if r["series_type"] == "MIP"] or \
                     [r for r in self.records if r["series_type"].startswith("T=")] or \
                     self.records
        if not candidates:
            messagebox.showerror("Error", "Pick a parent folder first.")
            return
        first = candidates[0]
        self._log(f"\nLoading for VOI placement: {first['series_folder']}")
        self._set_status("Loading volume...")
        vol = load_volume(first["series_folder"])
        if vol is None:
            messagebox.showerror("Error", "Could not load volume.")
            self._set_status("error.")
            return
        win = VoiSetupWindow(self.root, vol, self.config)
        self.root.wait_window(win.top)
        if win.committed:
            self.config = win.cfg
            self._log(f"VOI committed:  centre={self.config.voi_centre_mm}  size={self.config.voi_size_mm}  HU={self.config.threshold_hu}")
            self._set_status("config set.")
        else:
            self._log("VOI placement cancelled.")
            self._set_status("ready.")

    def on_save_cfg(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON","*.json")],
                                            title="Save config")
        if not path:
            return
        with open(path, "w") as f:
            f.write(self.config.to_json())
        self._log(f"Config saved: {path}")

    def on_load_cfg(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")],
                                          title="Load config")
        if not path:
            return
        try:
            with open(path) as f:
                self.config = AnalysisConfig.from_json(f.read())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")
            return
        self._log(f"Config loaded: {path}")
        self._log(f"  centre={self.config.voi_centre_mm}  size={self.config.voi_size_mm}  HU={self.config.threshold_hu}")

    def on_run_batch(self):
        if not self.records:
            messagebox.showerror("Error", "Pick a parent folder first.")
            return

        # Filter by selected types
        allowed = set()
        if self.var_mip.get():     allowed.add("MIP")
        if self.var_raw.get():     allowed.add("raw_4DCT")
        if self.var_unknown.get(): allowed.add("unknown")

        to_process = [r for r in self.records
                      if r["series_type"] in allowed
                      or (self.var_phase.get() and r["series_type"].startswith("T="))]

        if not to_process:
            messagebox.showwarning("Nothing to process",
                                   "No series match the selected types.\nCheck the checkboxes above.")
            return

        # Ask where to save QC images
        qc_dir = filedialog.askdirectory(
            title="Choose folder to save QC images (cancel to skip)"
        )
        save_qc = bool(qc_dir)
        if save_qc:
            os.makedirs(qc_dir, exist_ok=True)
            self._log(f"QC images will be saved to: {qc_dir}")
        else:
            self._log("QC images skipped.")

        self._log(f"\nProcessing {len(to_process)} series...")
        rows = []
        n    = len(to_process)

        for idx, rec in enumerate(to_process, 1):
            self._set_status(f"Processing {idx}/{n}: {os.path.basename(rec['series_folder'])}")
            self._log(f"[{idx}/{n}] ExamID={rec['exam_id']}  Type={rec['series_type']}  {rec['series_folder']}")
            try:
                vol = load_volume(rec["series_folder"])
                if vol is None:
                    self._log("  skipped: could not load volume")
                    continue
                result = analyse(vol, self.config)
                row    = result_to_row(rec, vol.series_meta, self.config, result)
                rows.append(row)
                self._log(f"  volume={result.volume_cc:.4f} cc  voxels={result.n_voxels_in_band}"
                          + (f"  notes: {result.notes}" if result.notes else ""))

                # Save QC image
                if save_qc:
                    try:
                        img_path = save_qc_image(vol, self.config, result, rec, qc_dir)
                        self._log(f"  QC image: {os.path.basename(img_path)}")
                    except Exception as qc_err:
                        self._log(f"  QC image failed: {qc_err}")

            except Exception as e:
                self._log(f"  ERROR: {e}")
                traceback.print_exc()

        self._set_status(f"Done: {len(rows)}/{n} rows.")
        self._log(f"\nBatch complete. {len(rows)} results.")
        if save_qc:
            self._log(f"QC images saved to: {qc_dir}")
        if rows:
            ResultsWindow(self.root, rows,
                          default_save_dir=self.parent_folder or os.getcwd())


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()