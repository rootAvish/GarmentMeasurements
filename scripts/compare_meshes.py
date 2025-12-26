import trimesh
import numpy as np

MESH_A_FILE = "manequin_clo3d_apose_male.obj"
MESH_B_FILE = "mean_female.obj"

try:
    mesh_a = trimesh.load_mesh(MESH_A_FILE)
    mesh_b = trimesh.load_mesh(MESH_B_FILE)

    # Get the size of the bounding box in each dimension (X, Y, Z)
    extents_a = mesh_a.extents
    extents_b = mesh_b.extents

    # Find the index of the largest dimension (0=X, 1=Y, 2=Z)
    up_axis_a = np.argmax(extents_a)
    up_axis_b = np.argmax(extents_b)
    
    axis_map = ['X', 'Y', 'Z']

    print(f"Mesh A Extents (X, Y, Z): {extents_a}")
    print(f"Mesh B Extents (X, Y, Z): {extents_b}")
    print("-" * 30)
    print(f"Mesh A 'up' axis (largest): {axis_map[up_axis_a]} (Size: {extents_a[up_axis_a]:.2f})")
    print(f"Mesh B 'up' axis (largest): {axis_map[up_axis_b]} (Size: {extents_b[up_axis_b]:.2f})")

    if up_axis_a == up_axis_b:
        print("\n✅ SUCCESS: Both meshes use the same 'up' axis.")
        print("You can proceed. The scale difference will be handled.")
    else:
        print(f"\n❌ PROBLEM: Meshes have different 'up' axes.")
        print(f"Mesh A is {axis_map[up_axis_a]}-up, but Mesh B is {axis_map[up_axis_b]}-up.")
        print("You must rotate one mesh to match the other before proceeding.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure 'trimesh' is installed (pip install trimesh) and file paths are correct.")
