import trimesh
import io

def load_geometric_mesh(file_path: str) -> trimesh.Trimesh:
    """
    Loads a .obj file as a "geometry-only" mesh.

    This function strips all data except for vertex positions ('v ')
    and face definitions ('f ') before loading. This forces trimesh
    to load the mesh with a vertex list that directly corresponds to
    the 'v' lines in the file, preserving the original vertex order
    and count, just as tools like Blender or PMP-library do.

    This is useful for ensuring vertex indices are consistent
    across different 3D libraries.

    Args:
        file_path: The file path to the .obj mesh.

    Returns:
        A trimesh.Trimesh object.
    """
    # Create an in-memory file object
    geo_only_file = io.StringIO()

    with open(file_path, 'r') as f:
        for line in f:
            # Keep only the vertex position ('v ') and face ('f ') lines
            if line.startswith('v ') or line.startswith('f '):
                geo_only_file.write(line)

    # Reset the in-memory file's "cursor" to the beginning
    geo_only_file.seek(0)

    # Load the mesh from the in-memory, geometry-only file
    mesh = trimesh.load_mesh(geo_only_file, file_type='obj')

    return mesh
