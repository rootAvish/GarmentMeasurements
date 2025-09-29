#!/usr/bin/env python3
"""
Retarget landmarks and measurement selections from the template mesh
to a target OBJ (e.g., a CLO3D avatar) by nearest-vertex mapping.

Inputs:
  - template OBJ (defaults to data/pca/mean.obj)
  - target OBJ (your avatar)
  - data dir (defaults to ./data)
  - output dir for retargeted files

Outputs:
  - <out_dir>/landmarks.json               (retargeted landmark vertex ids)
  - <out_dir>/measurements/*.mes           (retargeted selections; other fields preserved)

Notes:
  - This preserves measurement options, planes, and halfspaces.
  - Only the integer vertex indices in "selection" arrays are remapped.
  - Landmarks are remapped by finding nearest vertex to the template landmark position.
  - For meshes with different topology/density, duplicates can occur. The tool
    removes consecutive duplicates in selections to avoid degenerate segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_vertices_from_obj(path: Path) -> np.ndarray:
    """Load vertex positions (Nx3) from an OBJ file."""
    verts: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("v "):
                try:
                    _, xs, ys, zs = line.strip().split()[:4]
                    verts.append((float(xs), float(ys), float(zs)))
                except Exception:
                    # skip malformed vertex lines
                    continue
    if not verts:
        raise RuntimeError(f"No vertices found in OBJ: {path}")
    return np.asarray(verts, dtype=np.float64)


def map_points_to_nearest_indices(
    src_points: np.ndarray, target_vertices: np.ndarray
) -> List[int]:
    """Map each source 3D point to the nearest vertex index in target_vertices.

    Uses vectorized argmin per point. For large inputs this is memory-heavy but
    manageable for typical selection sizes.
    """
    mapped: List[int] = []
    # process in chunks to avoid large memory spikes
    chunk = 2048
    for i in range(0, len(src_points), chunk):
        batch = src_points[i : i + chunk]
        # distances: (batch_size, N)
        d2 = ((batch[:, None, :] - target_vertices[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        mapped.extend(idx.tolist())
    return mapped


def remove_consecutive_duplicates(ids: List[int]) -> List[int]:
    if not ids:
        return ids
    out = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            out.append(ids[i])
    return out


def retarget_landmarks(
    template_vertices: np.ndarray,
    target_vertices: np.ndarray,
    template_landmarks_path: Path,
) -> Dict[str, int]:
    data = json.loads(template_landmarks_path.read_text())
    lm = data.get("landmarks", {})
    names = list(lm.keys())
    ids = [int(lm[n]) for n in names]

    src_points = template_vertices[np.asarray(ids, dtype=np.int64), :]
    mapped = map_points_to_nearest_indices(src_points, target_vertices)
    return {name: int(idx) for name, idx in zip(names, mapped)}


def retarget_measurement_selection(
    measurement_json: dict,
    template_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> dict:
    sel = measurement_json.get("selection", [])
    if not sel:
        return measurement_json  # nothing to retarget

    src_points = template_vertices[np.asarray(sel, dtype=np.int64), :]
    mapped = map_points_to_nearest_indices(src_points, target_vertices)
    mapped = remove_consecutive_duplicates(mapped)
    mj = dict(measurement_json)
    mj["selection"] = mapped
    return mj


def main():
    ap = argparse.ArgumentParser(description="Retarget measurement data to a target OBJ")
    ap.add_argument("target_obj", type=Path, help="Target OBJ (e.g., CLO avatar)")
    ap.add_argument(
        "--template_obj",
        type=Path,
        default=Path("data/pca/mean.obj"),
        help="Template OBJ that matches original data selections",
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing landmarks_female.json and measurements/*.mes",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data_retargeted"),
        help="Output directory for retargeted files",
    )
    args = ap.parse_args()

    template_obj = args.template_obj
    target_obj = args.target_obj
    data_dir = args.data_dir
    out_dir = args.out_dir
    out_meas_dir = out_dir / "measurements"

    out_meas_dir.mkdir(parents=True, exist_ok=True)

    # Load meshes
    template_vertices = load_vertices_from_obj(template_obj)
    target_vertices = load_vertices_from_obj(target_obj)

    # Landmarks
    lm_in = data_dir / "landmarks_female.json"
    lm_out = out_dir / "landmarks.json"
    mapped_landmarks = retarget_landmarks(template_vertices, target_vertices, lm_in)
    lm_out.write_text(json.dumps({"landmarks": mapped_landmarks}, indent=2))

    # Measurements
    meas_dir = data_dir / "measurements"
    for p in sorted(meas_dir.glob("*.mes")):
        try:
            mj = json.loads(p.read_text())
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        mj2 = retarget_measurement_selection(mj, template_vertices, target_vertices)
        (out_meas_dir / p.name).write_text(json.dumps(mj2, indent=2))

    print(f"Retargeted landmarks -> {lm_out}")
    print(f"Retargeted measurements -> {out_meas_dir}")
    print("Next: run measurements with --data_dir pointing to out_dir")


if __name__ == "__main__":
    main()

