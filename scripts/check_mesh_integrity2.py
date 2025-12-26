import trimesh
import sys
import numpy as np
import logging

# Configure logging to show trimesh messages if any issues occur during loading
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
obj_file_path = sys.argv[1]


# --- Load Mesh ---
print(f"Loading mesh from: {obj_file_path}")
try:
    loaded_data = trimesh.load(obj_file_path, process=False)

    if isinstance(loaded_data, trimesh.Scene):
        print("Loaded data is a Scene object. Attempting to extract main geometry.")
        # Try combining all meshes into one
        mesh = loaded_data.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
             raise TypeError("Could not extract a single Trimesh object from the Scene.")
        print(f"Successfully extracted mesh geometry from Scene.")
    elif isinstance(loaded_data, trimesh.Trimesh):
        mesh = loaded_data
        print(f"Successfully loaded mesh.")
    else:
        raise TypeError(f"Loaded data is an unexpected type: {type(loaded_data)}")

    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")

except Exception as e:
    print(f"Error loading or processing mesh: {e}")
    exit()
original_vertex_count = len(mesh.vertices)


# 4. Duplicate Vertices
print("\nChecking for duplicate vertices (at the same position)...")
try:
    print(f"Vertices are: {mesh.vertices}")
    unique_vertices, inverse_indices = trimesh.grouping.unique_rows(mesh.vertices)
    print(unique_vertices)
    merged_vertex_count = len(unique_vertices)
    duplicates_found = original_vertex_count - merged_vertex_count
    if duplicates_found == 0:
        print("  Result: No duplicate vertices found. ✅")
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices[unique_vertices],  # The new, small vertex list
        faces=inverse_indices[mesh.faces]       # The faces, re-indexed
    )
except Exception as e:
    print(f"  Error during vertex duplicate check: {e}")


components = mesh.split(only_watertight=False)

print(f"Found {len(components)} components.")

for i, comp in enumerate(components):
    if len(comp.faces) <= 1:
        print(f"\n--- Component {i} (Artifact?) ---")
        print(f"  Faces: {len(comp.faces)}")
        print(f"  Vertices: {len(comp.vertices)}")
        # Print the actual coordinates to see where they are
        print(f"  Location (Bounding Box Center): {comp.bounds.mean(axis=0)}")
        # print(f"  Vertex Coordinates:\n{comp.vertices}")

# print(f"--- Cleaning components ---")
# print(f"Original mesh had {len(mesh.faces)} faces and {len(mesh.split(only_watertight=False))} components.")

# # 1. Split the mesh into its 8 components
# components = mesh.split(only_watertight=False)

# # 2. Find the largest component by checking the face count of each
# if components: # Ensure list is not empty
#     largest_component = max(components, key=lambda comp: len(comp.faces))

#     # 3. Overwrite your 'mesh' variable with *only* the largest part
#     mesh = largest_component

#     print(f"Cleaned mesh now has {len(mesh.faces)} faces and is a single component. ✅")
# else:
#     print("Mesh splitting resulted in no components.")

num_vertices = len(mesh.vertices)

# 5. Disconnected Components & Manifoldness (via split)
print("\nChecking for disconnected components (and implicitly manifoldness)...")
num_components = 0 # Default to 0, indicating an issue or single component
components = []
try:
    # split() often requires manifold or fixes it internally.
    # only_watertight=False allows splitting non-watertight meshes.
    components = mesh.split(only_watertight=False)
    num_components = len(components)
    if num_components == 1:
        # If it successfully split into 1 component, it implies it was manifold enough for the operation.
        print("  Result: Mesh is a single connected component. ✅")
        # Check watertightness again *after* split, as split might repair things.
        if not mesh.is_watertight:
             print("    Note: Mesh became watertight after internal processing by split(), or was already watertight. ✅")
        else:
             print("    Mesh remains watertight after split(). ✅")

    else:
        print(f"  Result: Mesh has {num_components} disconnected components. ❌")
        component_faces = [len(comp.faces) for comp in components]
        print(f"    Component sizes (by face count): {sorted(component_faces, reverse=True)}")

except Exception as e:
    print(f"  Error during component splitting: {e} ❌")
    print(f"    This often indicates the mesh is non-manifold or has complex issues.")
    num_components = -1 # Use -1 to indicate failure

# mesh.export('clean_mesh.obj')

print("Cleaned mesh successfully saved to 'clean_mesh.obj'")

sys.exit(0)
# --- Perform Integrity Checks ---
print("\n--- Running Mesh Integrity Checks ---")

duplicates_found = 0
original_vertex_count = len(mesh.vertices)

# 1. Watertight & Manifold Check
#    Check watertightness initially. We'll rely on split() later for manifold info.
print(f"\nChecking if mesh is initially watertight...")
is_initially_watertight = mesh.is_watertight
if is_initially_watertight:
    print("  Result: Mesh appears initially watertight. ✅")
else:
    print("  Result: Mesh is NOT initially watertight. ❌")
    try:
        boundary_edges = mesh.edges_unique[mesh.edges_unique_length == 1]
        print(f"    Number of boundary edges found: {len(boundary_edges)}")
    except Exception as e:
         print(f"    Could not determine initial boundary edges: {e}")

# --- >>> MODIFIED: Removed direct mesh.is_manifold call <<< ---

# 2. Face Winding Consistency
print(f"\nChecking face winding consistency...")
if mesh.is_winding_consistent:
     print("  Result: Face windings are consistent. ✅")
else:
     print("  Result: Face windings are potentially inconsistent. ❌")

# 3. Duplicate Faces
print("\nChecking for duplicate faces...")
try:
    if mesh.faces.shape[1] == 3:
        if hasattr(mesh, 'faces_sparse') and callable(getattr(mesh, 'faces_sparse', None)):
             if mesh.faces_sparse.sum() == len(mesh.faces):
                 print("  Result: No duplicate faces found (using sparse check). ✅")
             else:
                  print(f"  Result: Found {len(mesh.faces) - mesh.faces_sparse.sum()} duplicate faces (using sparse check). ❌")
        else: # Fallback if faces_sparse not available
            unique_faces, inverse_indices = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_inverse=True)
            if len(unique_faces) == len(mesh.faces):
                print("  Result: No duplicate faces found (using unique check). ✅")
            else:
                print(f"  Result: Found {len(mesh.faces) - len(unique_faces)} duplicate faces (using unique check). ❌")
    else:
        unique_faces, inverse_indices = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_inverse=True)
        if len(unique_faces) == len(mesh.faces):
            print("  Result: No duplicate faces found (non-triangular, unique check). ✅")
        else:
            print(f"  Result: Found {len(mesh.faces) - len(unique_faces)} duplicate faces (non-triangular, unique check). ❌")
except Exception as e:
    print(f"  Could not perform duplicate face check: {e}")
