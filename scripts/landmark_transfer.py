"""
Transfers landmarks from a source mesh (Mesh A) to a target mesh (Mesh B).

This script works by:
1.  Normalizing both meshes (centering at origin, scaling to 1.0) to
    ensure they are in a comparable space.
2.  Loading the source landmarks (Mesh A) and finding their coordinates
    on the *normalized* Mesh A.
3.  Building a k-d tree from the vertices of the *normalized* Mesh B.
4.  Querying this k-d tree to find the nearest vertex on normalized Mesh B
    for each landmark from normalized Mesh A.
5.  Saving the indices of these nearest neighbors as the new landmarks for Mesh B.

The final JSON file contains landmark names mapped to vertex indices of the
*original* Mesh B.
"""

import trimesh
import json
from scipy.spatial import cKDTree
from trimesh_utils import load_geometric_mesh
import numpy as np
import argparse # Added for CLI arguments

def transfer_landmarks_aligned(mesh_a_path, mesh_b_path, landmarks_a_path, landmarks_b_path):
    """
    Loads, aligns, and transfers landmarks from mesh A to mesh B.

    Args:
        mesh_a_path (str): File path to the source mesh (e.g., .obj)
        mesh_b_path (str): File path to the target mesh (e.g., .obj)
        landmarks_a_path (str): File path to the source JSON landmarks file.
        landmarks_b_path (str): File path to *write* the new target JSON landmarks.
    """
    try:
        print(f"Loading Mesh A: {mesh_a_path}")
        mesh_a = load_geometric_mesh(mesh_a_path)

        print(f"Loading Mesh B: {mesh_b_path}")
        mesh_b = load_geometric_mesh(mesh_b_path)

        print(f"Loading Landmarks A: {landmarks_a_path}")
        with open(landmarks_a_path, 'r') as f:
            landmarks_a_data = json.load(f)

    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please ensure trimesh and numpy are installed and file paths are correct.")
        return

    # --- 2. Alignment Step (Normalize Meshes) ---
    # We will work with copies so we don't modify the originals
    print("Normalizing meshes for alignment...")
    mesh_a_aligned = mesh_a.copy()
    mesh_b_aligned = mesh_b.copy()

    # Move Mesh A to origin (0,0,0) and scale its longest side to 1.0
    mesh_a_aligned.apply_translation(-mesh_a_aligned.centroid)
    mesh_a_aligned.apply_scale(1.0 / mesh_a_aligned.scale)

    # Move Mesh B to origin (0,0,0) and scale its longest side to 1.0
    mesh_b_aligned.apply_translation(-mesh_b_aligned.centroid)
    mesh_b_aligned.apply_scale(1.0 / mesh_b_aligned.scale)

    # Now, both meshes are at the same position and scale,
    # making the nearest-neighbor search meaningful.

    # --- 3. Get Landmark Coordinates from Aligned Mesh A ---

    # Get the original landmark indices
    try:
        landmark_names = list(landmarks_a_data["landmarks"].keys())
        landmark_indices_a = list(landmarks_a_data["landmarks"].values())

        # Get the (x, y, z) coordinates of the landmarks *from the aligned mesh*
        landmark_coords_a = mesh_a_aligned.vertices[landmark_indices_a]
    except KeyError:
        print("Error: Landmarks JSON for Mesh A must be in the format {'landmarks': {'name': index, ...}}")
        return
    except IndexError:
        print("Error: A landmark index in the JSON for Mesh A is out of bounds for the mesh vertices.")
        return


    # --- 4. Build KD-Tree for Aligned Mesh B ---
    print("Building KD-Tree for Aligned Mesh B...")
    tree_b = cKDTree(mesh_b_aligned.vertices)

    # --- 5. Transfer Landmarks ---
    new_landmarks = {}
    print("Transferring landmarks...")

    # Query the tree for all landmarks at once
    distances, new_indices_b = tree_b.query(landmark_coords_a)

    for i, name in enumerate(landmark_names):
        index_a = landmark_indices_a[i]
        index_b = new_indices_b[i]
        distance = distances[i]

        # Store the new index
        new_landmarks[name] = int(index_b) # Ensure it's a standard Python int for JSON

        print(f"  - '{name}': Index {index_a} (Mesh A) -> Index {index_b} (Mesh B) (Dist: {distance:.4f})")

    # --- 6. Save the New JSON File ---
    # The new indices refer to the *original* Mesh B,
    # as we only used mesh_b_aligned for the calculation.
    output_data = {"landmarks": new_landmarks}

    try:
        with open(landmarks_b_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"\nSuccessfully created new landmark file: {landmarks_b_path}")
    except IOError as e:
        print(f"Error writing output file {landmarks_b_path}: {e}")


# --- Run the script ---
if __name__ == "__main__":
    # --- 1. Set up CLI Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Transfer 3D mesh landmarks from a source mesh (A) to a target mesh (B) using nearest neighbors on normalized meshes.",
        formatter_class=argparse.RawTextHelpFormatter # Keeps help text formatting
    )

    parser.add_argument(
        "--meshA",
        metavar="PATH_A_MESH",
        type=str,
        required=True,
        help="Path to the source mesh file (e.g., mean_female.obj)."
    )

    parser.add_argument(
        "--meshB",
        metavar="PATH_B_MESH",
        type=str,
        required=True,
        help="Path to the target mesh file (e.g., manequin_clo3d.obj)."
    )

    parser.add_argument(
        "--landmarksA",
        metavar="PATH_A_JSON",
        type=str,
        required=True,
        help="Path to the source landmarks JSON file (e.g., landmarks_female.json)."
    )

    parser.add_argument(
        "--landmarksB",
        metavar="PATH_B_JSON_OUT",
        type=str,
        required=True,
        help="Path for the *output* transferred landmarks JSON file (e.g., landmarks_clo_manequin.json)."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided arguments
    transfer_landmarks_aligned(
        args.meshA,
        args.meshB,
        args.landmarksA,
        args.landmarksB
    )
