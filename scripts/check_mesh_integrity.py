import trimesh
import json
import numpy as np
import logging

# Configure logging to show trimesh messages if any issues occur during loading
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
obj_file_path = "./mean_female.obj"
landmarks_file_path = "./data/landmarks_female.json" # Make sure this file exists and contains your landmark data

# Landmarks relevant for the arm_length check (1-based from your JSON)
wrist_landmark_name = "wrist_right"
shoulder_landmark_name = "shoulder_right"

# --- Load Mesh ---
print(f"Loading mesh from: {obj_file_path}")
try:
    # process=False avoids some automatic checks/fixes during loading,
    # so we can perform them explicitly.
    mesh = trimesh.load(obj_file_path, process=False)
    print(f"Successfully loaded mesh.")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
except Exception as e:
    print(f"Error loading mesh: {e}")
    exit()

# --- Load Landmarks ---
print(f"\nLoading landmarks from: {landmarks_file_path}")
try:
    with open(landmarks_file_path, 'r') as f:
        landmarks_data = json.load(f)
    # Convert 1-based indices from JSON to 0-based for trimesh
    # Assuming the structure is {"landmarks": {"name": index, ...}}
    landmarks = {name: index - 1 for name, index in landmarks_data.get("landmarks", {}).items()}
    print("Successfully loaded landmarks.")

    # Verify required landmarks exist
    if wrist_landmark_name not in landmarks:
        print(f"Error: Landmark '{wrist_landmark_name}' not found in landmarks file.")
        exit()
    if shoulder_landmark_name not in landmarks:
        print(f"Error: Landmark '{shoulder_landmark_name}' not found in landmarks file.")
        exit()

    wrist_vertex_index = landmarks[wrist_landmark_name]
    shoulder_vertex_index = landmarks[shoulder_landmark_name]
    print(f"  Using '{wrist_landmark_name}' index: {wrist_vertex_index} (0-based)")
    print(f"  Using '{shoulder_landmark_name}' index: {shoulder_vertex_index} (0-based)")

    # Verify indices are within bounds
    num_vertices = len(mesh.vertices)
    if not (0 <= wrist_vertex_index < num_vertices):
         print(f"Error: Wrist index {wrist_vertex_index} out of bounds (0-{num_vertices-1}).")
         exit()
    if not (0 <= shoulder_vertex_index < num_vertices):
         print(f"Error: Shoulder index {shoulder_vertex_index} out of bounds (0-{num_vertices-1}).")
         exit()

except Exception as e:
    print(f"Error loading or processing landmarks: {e}")
    exit()


# --- Perform Integrity Checks ---
print("\n--- Running Mesh Integrity Checks ---")

# 1. Watertight & Manifold Check
#    is_watertight implies is_manifold and no boundary edges.
print(f"\nChecking if mesh is watertight (manifold, enclosed)...")
if mesh.is_watertight:
    print("  Result: Mesh is watertight. ✅")
else:
    print("  Result: Mesh is NOT watertight. ❌")
    # Provide more details if not watertight
    print(f"    Is manifold: {mesh.is_manifold}")
    boundary_edges = mesh.edges_unique[mesh.edges_unique_length == 1]
    print(f"    Number of boundary edges: {len(boundary_edges)}")
    if mesh.nonmanifold_edges is not None and len(mesh.nonmanifold_edges) > 0:
        print(f"    Number of non-manifold edges: {len(mesh.nonmanifold_edges)}")


# 2. Face Winding Consistency (Normals direction)
print(f"\nChecking face winding consistency...")
if mesh.is_winding_consistent:
     print("  Result: Face windings are consistent. ✅")
else:
     print("  Result: Face windings are potentially inconsistent. ❌")
     # Optional: Attempt to fix winding and recheck
     # mesh.fix_normals()
     # print(f"  Attempted fix. Rechecking: {mesh.is_winding_consistent}")

# 3. Duplicate Faces
print("\nChecking for duplicate faces...")
# faces_sparse uses a sparse matrix to quickly find duplicates
if mesh.faces_sparse.sum() == len(mesh.faces):
    print("  Result: No duplicate faces found. ✅")
else:
    # A more thorough check (might be slower on huge meshes)
    unique_faces, inverse_indices = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_inverse=True)
    if len(unique_faces) == len(mesh.faces):
        print("  Result: No duplicate faces found (confirmed with slower check). ✅")
    else:
        print(f"  Result: Found {len(mesh.faces) - len(unique_faces)} duplicate faces. ❌")


# 4. Duplicate Vertices (based on position, using trimesh's built-in merge)
print("\nChecking for duplicate vertices (at the same position)...")
try:
    # Use merge_vertices which is designed for this. It modifies the mesh in place!
    # If you don't want to modify, you'd need a different approach.
    original_vertex_count = len(mesh.vertices)
    mesh.merge_vertices()
    merged_vertex_count = len(mesh.vertices)
    duplicates_found = original_vertex_count - merged_vertex_count
    if duplicates_found == 0:
        print("  Result: No duplicate vertices found. ✅")
    else:
        # NOTE: This merge invalidates original landmark indices if duplicates existed!
        print(f"  Result: Found and merged {duplicates_found} duplicate vertices. ❌")
        print("          IMPORTANT: Landmark indices may now be incorrect!")
        # If duplicates were merged, you might need to remap landmarks based on the new vertices.
        # This requires knowing the merge mapping, which trimesh doesn't easily expose after merge_vertices.
        # Consider using `trimesh.grouping.group_rows(mesh.vertices, digits=...)` before merging
        # if you need to precisely track merged indices.

except Exception as e:
    print(f"  Error during vertex merge check: {e}")

# Reset vertex count in case merge happened for subsequent checks
num_vertices = len(mesh.vertices)
# Re-validate landmark indices if merge occurred and indices might be invalid
if duplicates_found > 0:
    print("  Re-validating landmark indices after vertex merge...")
    if not (0 <= wrist_vertex_index < num_vertices):
         print(f"  Error: Original wrist index {wrist_vertex_index} is now out of bounds (0-{num_vertices-1}) due to merge.")
         # Cannot proceed with path check
         wrist_vertex_index = -1 # Mark as invalid
    if not (0 <= shoulder_vertex_index < num_vertices):
         print(f"  Error: Original shoulder index {shoulder_vertex_index} is now out of bounds (0-{num_vertices-1}) due to merge.")
          # Cannot proceed with path check
         shoulder_vertex_index = -1 # Mark as invalid

# 5. Disconnected Components
print("\nChecking for disconnected components...")
# Calculate face adjacency graph
components = mesh.split(only_watertight=False)
num_components = len(components)
if num_components == 1:
    print("  Result: Mesh is a single connected component. ✅")
else:
    print(f"  Result: Mesh has {num_components} disconnected components. ❌")
    # You could optionally save each component to inspect it:
    # for i, part in enumerate(components):
    #     part.export(f"component_{i}.obj")

# --- Geodesic Path Check (Arm Length Landmarks) ---
print("\n--- Checking Geodesic Path ---")
print(f"Finding shortest path between '{wrist_landmark_name}' ({wrist_vertex_index}) and '{shoulder_landmark_name}' ({shoulder_vertex_index})...")

if wrist_vertex_index == -1 or shoulder_vertex_index == -1:
    print("  Skipping path check due to invalid landmark indices after vertex merge.")
else:
    try:
        # Get the adjacency graph for edges
        edges = mesh.edges_unique
        length = mesh.edges_unique_length
        # Build the graph (this finds vertex neighbors)
        g = trimesh.graph.edges_to_coo(edges, length)

        # Run the shortest path query
        path_nodes, path_distance = trimesh.graph.shortest_path(
            graph=g,
            start_node=wrist_vertex_index,
            end_node=shoulder_vertex_index
        )

        if path_nodes is not None and len(path_nodes) > 0:
            print(f"  Result: Path found! ✅")
            print(f"    Distance: {path_distance} (in mesh units, likely meters)")
            print(f"    Number of vertices in path: {len(path_nodes)}")
            # The extreme value you saw (3.4e+40) likely came from the C++ library
            # returning a very large float (like FLT_MAX) when no path was found.
        else:
            print(f"  Result: No path found between the specified landmarks. ❌")
            print("          This is the likely reason for the infinite 'arm_length'.")
            print("          Possible causes: Landmarks are on disconnected components, mesh errors along the path.")

    except Exception as e:
        print(f"  Error during shortest path calculation: {e}")
        print("          This could also indicate mesh connectivity issues or numerical problems.")

print("\n--- Checks Complete ---")
