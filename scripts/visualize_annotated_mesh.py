import trimesh
import plotly.graph_objects as go
import json
import numpy as np

# --- 1. Configuration ---

# !!! IMPORTANT !!!
# Change this to the path of your .obj file
MESH_FILE_PATH = 'manequin_clo3d_apose_male.obj'

LANDMARK_FILE_PATH = 'landmarks_clo_manequin.json'

# --- 2. Load Data ---

print(f"Loading mesh from {MESH_FILE_PATH}...")
try:
    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(MESH_FILE_PATH)
    print("Mesh loaded successfully.")
except Exception as e:
    print(f"Error loading mesh '{MESH_FILE_PATH}': {e}")
    print("Please make sure the file path is correct.")
    exit()

print(f"Loading landmarks from {LANDMARK_FILE_PATH}...")
try:
    # Load the landmark JSON
    with open(LANDMARK_FILE_PATH) as f:
        landmark_data = json.load(f)
    print("Landmarks loaded successfully.")
except Exception as e:
    print(f"Error loading JSON '{LANDMARK_FILE_PATH}': {e}")
    exit()

# --- 3. Prepare Landmark Data ---

landmarks_dict = landmark_data['landmarks']

# Get the names (e.g., "nape_of_neck")
landmark_names = list(landmarks_dict.keys())
# Get the indices (e.g., 30197)
landmark_indices = list(landmarks_dict.values())

# Get the 3D coordinates (x, y, z) from the mesh for each landmark index
# mesh.vertices is a NumPy array of all vertex coordinates
landmark_coords = mesh.vertices[landmark_indices]

# Split landmark coordinates for Plotly
lx = landmark_coords[:, 0]
ly = landmark_coords[:, 1]
lz = landmark_coords[:, 2]

# --- 4. Create Visualization Traces ---

# Create the 3D mesh trace for Plotly
# This uses the vertices and faces from the loaded mesh
mesh_trace = go.Mesh3d(
    x=mesh.vertices[:, 0],
    y=mesh.vertices[:, 1],
    z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0],
    j=mesh.faces[:, 1],
    k=mesh.faces[:, 2],
    color='lightgray',  # Neutral color for the mesh
    opacity=0.6,        # Make it semi-transparent to see points behind
    name='Mannequin Mesh'
)

# Create the 3D scatter plot trace for the landmarks
landmark_trace = go.Scatter3d(
    x=lx,
    y=ly,
    z=lz,
    mode='markers+text',  # Show both a marker and the text label
    text=landmark_names,
    textposition='top center', # Position the text label
    marker=dict(
        color='red',
        size=5  # Adjust marker size as needed
    ),
    name='Landmarks'
)

# Combine the mesh and landmark traces into one dataset
plot_data = [mesh_trace, landmark_trace]

# --- 5. Plot Front View ---

# Define the camera for a front view

camera_front = dict(
    up=dict(x=0, y=1, z=0),      # <-- FIX: Y-axis is "up"
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0.25, z=-2.0) # <-- FIX: Position camera in front (negative Z)
                                 # and slightly elevated (positive Y)
    # Adjust the 'z' value (e.g., -2.0) to zoom in/out
)

fig_front = go.Figure(data=plot_data)
fig_front.update_layout(
    title='Front View with Landmarks (Y-Up)',
    scene_camera=camera_front,
    scene=dict(
        xaxis_title='X (Width)',
        yaxis_title='Y (Height)',  # <-- FIX: Y is Height
        zaxis_title='Z (Depth)',   # <-- FIX: Z is Depth
        aspectmode='data'      # Ensures the mesh is not stretched
    )
)

print("Displaying Front View...")
fig_front.show()

# --- 6. Plot Back View ---

# Define the camera for a back view
camera_back = dict(
    up=dict(x=0, y=1, z=0),      # <-- FIX: Y-axis is "up"
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0.25, z=2.0) # <-- FIX: Position camera behind (positive Z)
)
fig_back = go.Figure(data=plot_data)
fig_back.update_layout(
    title='Back View with Landmarks (Y-Up)',
    scene_camera=camera_back,
    scene=dict(
        xaxis_title='X (Width)',
        yaxis_title='Y (Height)',  # <-- FIX: Y is Height
        zaxis_title='Z (Depth)',   # <-- FIX: Z is Depth
        aspectmode='data'      # Ensures the mesh is not stretched
    )
)

print("Displaying Back View...")
fig_back.show()
