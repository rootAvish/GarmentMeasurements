#!/usr/bin/env python3
"""
Remap selection vertex indices in .mes files from a source (template) mesh
to a target (avatar) mesh by nearest-vertex mapping.

Usage:
  python scripts/remap_mes_selection.py \
    --src-mesh /path/to/mean_female.obj \
    --dst-mesh /path/to/avatar.obj \
    --in-dir data/measurements \
    --out-dir data/measurements_manequin \

Notes:
  - Keeps all fields in the .mes files untouched except `selection`.
  - Order is preserved with stable de-duplication.
  - Skips invalid indices and warns.
  - Both meshes are translated to origin (by centroid) and uniformly scaled to
    unit longest side before nearest mapping, mirroring the landmark transfer.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import trimesh
from trimesh_utils import load_geometric_mesh
from scipy.spatial import cKDTree



def vertices_aligned(mesh: trimesh.Trimesh) -> np.ndarray:
    # Work on a copy to avoid mutating the original
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    m.apply_scale(1.0 / float(m.scale))
    return np.asarray(m.vertices)


def get_vertices_aligned(mesh: trimesh.Trimesh) -> np.ndarray:
    return vertices_aligned(mesh)


def stable_unique(seq: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def axis_index(axis_str: str) -> int:
    s = axis_str.strip().upper()
    if s in {"X", "+X", "-X"}:
        return 0
    if s in {"Y", "+Y", "-Y"}:
        return 1
    return 2


def axis_normal(axis_str: str) -> np.ndarray:
    s = axis_str.strip().upper()
    v = np.zeros(3, dtype=float)
    if s in {"X", "+X"}:
        v[0] = 1.0
    elif s == "-X":
        v[0] = -1.0
    elif s in {"Y", "+Y"}:
        v[1] = 1.0
    elif s == "-Y":
        v[1] = -1.0
    elif s in {"Z", "+Z"}:
        v[2] = 1.0
    elif s == "-Z":
        v[2] = -1.0
    return v


def load_landmarks(path: Path) -> Dict[str, int]:
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: int(v) for k, v in data.get('landmarks', {}).items()}


def measurement_type_requires_selection(mtype: str) -> bool:
    # Only these rely on `selection` seeds for candidate faces
    return mtype in {"AxisAlignedPlane", "AxisAlignedPlanePart", "VertexTrace"}


def build_plane_from_landmarks(meas: dict, dst_landmarks: Dict[str, int], dst_vertices_aligned: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (point, normal) for measurement_plane if type == LANDMARKS, else None."""
    plane = meas.get('measurement_plane', {})
    if plane.get('type') != 'LANDMARKS':
        return None
    lm_names = plane.get('landmarks', [])
    second_axis = plane.get('landmarks_second_axis', 'Z')
    normal_axis = plane.get('normal_axis', 'Y')
    if len(lm_names) == 1:
        name = lm_names[0]
        if name not in dst_landmarks:
            return None
        pid = dst_landmarks[name]
        p = dst_vertices_aligned[pid]
        n = axis_normal(normal_axis)
        return p, n
    if len(lm_names) == 2:
        n0, n1 = lm_names
        if n0 not in dst_landmarks or n1 not in dst_landmarks:
            return None
        p0 = dst_vertices_aligned[dst_landmarks[n0]]
        p1 = dst_vertices_aligned[dst_landmarks[n1]]
        v01 = p1 - p0
        v01 = v01 / (np.linalg.norm(v01) + 1e-12)
        sec = axis_normal(second_axis)
        # plane normal = cross(v01, sec)
        n = np.cross(v01, sec)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n = n / n_norm
        else:
            n = axis_normal(normal_axis)
        return p0, n
    return None


def signed_distance_to_plane(p: np.ndarray, plane_p: np.ndarray, plane_n: np.ndarray) -> float:
    return float(np.dot(p - plane_p, plane_n))


def map_selection(src_vertices: np.ndarray,
                  dst_vertices: np.ndarray,
                  selection: List[int],
                  meas: dict,
                  dst_landmarks: Dict[str, int],
                  band: float = 0.02) -> List[int]:
    """Nearest mapping with landmark/plane guidance to avoid collapsing selections.

    - If measurement_plane.type == CENTROID: restrict candidates to a slab around the
      source mean along the normal axis.
    - If measurement_plane.type == LANDMARKS: build plane on target landmarks and
      restrict candidates to a slab around that plane.
    - Else: fall back to full-mesh KDTree.
    """
    n_src = src_vertices.shape[0]
    valid_indices = [idx for idx in selection if 0 <= idx < n_src]
    invalid = [idx for idx in selection if idx not in valid_indices]
    for idx in invalid:
        print(f"[WARN] selection index {idx} out of range for source mesh (n={n_src}); skipping")

    if not valid_indices:
        return []

    pts = src_vertices[np.asarray(valid_indices, dtype=np.int64)]

    # Candidate filtering
    plane = meas.get('measurement_plane', {})
    candidates = None
    if plane.get('type') == 'CENTROID':
        ax = axis_index(plane.get('normal_axis', 'Y'))
        target_val = float(np.mean(pts[:, ax]))
        mask = np.abs(dst_vertices[:, ax] - target_val) <= band
        candidates = np.where(mask)[0]
    elif plane.get('type') == 'LANDMARKS':
        built = build_plane_from_landmarks(meas, dst_landmarks, dst_vertices)
        if built is not None:
            plane_p, plane_n = built
            dists = np.abs(np.dot(dst_vertices - plane_p, plane_n))
            candidates = np.where(dists <= band)[0]

    # Build KDTree over candidates or full mesh
    if candidates is not None and candidates.size > 0:
        tree = cKDTree(dst_vertices[candidates])
        dists, mapped_local = tree.query(pts)
        remapped = [int(candidates[int(j)]) for j in mapped_local]
    else:
        tree = cKDTree(dst_vertices)
        dists, mapped = tree.query(pts)
        remapped = [int(j) for j in mapped]

    # Do not deduplicate; keep order to preserve distribution
    return remapped


def process_mes_file(src_vertices: np.ndarray,
                     dst_vertices: np.ndarray,
                     dst_landmarks: Dict[str, int],
                     in_path: Path,
                     out_path: Path) -> None:
    with open(in_path, 'r') as f:
        data = json.load(f)

    mtype = data.get('type', 'NONE')
    selection = data.get('selection', [])

    # If measurement doesn't require selection, copy through unchanged
    if not measurement_type_requires_selection(mtype):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[OK] copied {out_path} (type {mtype}, no selection remap)")
        return

    if not isinstance(selection, list) or len(selection) == 0:
        print(f"[WARN] {in_path}: selection missing/empty for type {mtype}; copying unchanged")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        return

    remapped = map_selection(src_vertices, dst_vertices, selection, data, dst_landmarks)
    if not remapped:
        print(f"[WARN] {in_path}: remapped selection is empty; copying unchanged")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        return

    data['selection'] = remapped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] wrote {out_path} (selection size {len(remapped)})")


def main():
    ap = argparse.ArgumentParser(description="Remap .mes selections by nearest vertices (landmark-guided)")
    ap.add_argument('--src-mesh', type=Path, required=True, help='Template/source mesh (mean_female)')
    ap.add_argument('--dst-mesh', type=Path, required=True, help='Target/avatar mesh')
    ap.add_argument('--in-dir', type=Path, required=True, help='Directory containing input .mes files')
    ap.add_argument('--out-dir', type=Path, required=True, help='Directory to write remapped .mes files')
    ap.add_argument('--src-landmarks', type=Path, default=Path('data/landmarks_female.json'), help='Source landmarks JSON (for context only)')
    ap.add_argument('--dst-landmarks', type=Path, default=Path('data/landmarks_clo_manequin.json'), help='Target landmarks JSON used to build planes')
    ap.add_argument('--only', type=str, nargs='*', default=None, help='Subset of .mes basenames to process')
    args = ap.parse_args()

    src_mesh = load_geometric_mesh(args.src_mesh)
    dst_mesh = load_geometric_mesh(args.dst_mesh)
    # Always use aligned vertices for mapping
    src_vertices = get_vertices_aligned(src_mesh)
    dst_vertices = get_vertices_aligned(dst_mesh)

    in_dir = args.in_dir
    out_dir = args.out_dir

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    mes_files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix == '.mes']
    if args.only:
        names = set(args.only)
        mes_files = [p for p in mes_files if p.name in names]

    if not mes_files:
        print(f"[INFO] No .mes files found in {in_dir}")
        return

    # Load landmarks for target mesh
    try:
        dst_landmarks = load_landmarks(args.dst_landmarks)
    except Exception as e:
        print(f"[WARN] Could not load dst landmarks {args.dst_landmarks}: {e}")
        dst_landmarks = {}

    print(f"[INFO] Processing {len(mes_files)} files…")
    for in_path in mes_files:
        out_path = out_dir / in_path.name
        try:
            process_mes_file(src_vertices, dst_vertices, dst_landmarks, in_path, out_path)
        except Exception as e:
            print(f"[ERR] {in_path}: {e}")


if __name__ == '__main__':
    main()
