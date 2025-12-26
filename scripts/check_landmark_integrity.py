# landmarks_file_path = "./data_manequin/landmarks_clo_manequin.json" # Make sure this file exists
# Landmarks relevant for the arm_length check (1-based from your JSON)
wrist_landmark_name = "wrist_right"
shoulder_landmark_name = "shoulder_right"

# --- Load Landmarks ---
# print(f"\nLoading landmarks from: {landmarks_file_path}")
# try:
#     with open(landmarks_file_path, 'r') as f:
#         landmarks_data = json.load(f)
#     landmarks = {name: index - 1 for name, index in landmarks_data.get("landmarks", {}).items()}
#     print("Successfully loaded landmarks.")

#     if wrist_landmark_name not in landmarks:
#         print(f"Error: Landmark '{wrist_landmark_name}' not found.")
#         exit()
#     if shoulder_landmark_name not in landmarks:
#         print(f"Error: Landmark '{shoulder_landmark_name}' not found.")
#         exit()

#     wrist_vertex_index = landmarks[wrist_landmark_name]
#     shoulder_vertex_index = landmarks[shoulder_landmark_name]
#     print(f"  Using '{wrist_landmark_name}' index: {wrist_vertex_index} (0-based)")
#     print(f"  Using '{shoulder_landmark_name}' index: {shoulder_vertex_index} (0-based)")

#     num_vertices = len(mesh.vertices)
#     if not (0 <= wrist_vertex_index < num_vertices):
#          print(f"Error: Wrist index {wrist_vertex_index} out of bounds (0-{num_vertices-1}).")
#          exit()
#     if not (0 <= shoulder_vertex_index < num_vertices):
#          print(f"Error: Shoulder index {shoulder_vertex_index} out of bounds (0-{num_vertices-1}).")
#          exit()

# except Exception as e:
#     print(f"Error loading or processing landmarks: {e}")
#     exit()


# if not (0 <= wrist_vertex_index < num_vertices):
#      print(f"  Error: Wrist index {wrist_vertex_index} is out of bounds (0-{num_vertices-1}).")
#      wrist_vertex_index = -1
# if not (0 <= shoulder_vertex_index < num_vertices):
#      print(f"  Error: Shoulder index {shoulder_vertex_index} is out of bounds (0-{num_vertices-1}).")
#      shoulder_vertex_index = -1


# # --- Geodesic Path Check ---
# print("\n--- Checking Geodesic Path ---")
# print(f"Finding shortest path between '{wrist_landmark_name}' ({wrist_vertex_index}) and '{shoulder_landmark_name}' ({shoulder_vertex_index})...")

# path_possible = True
# if wrist_vertex_index == -1 or shoulder_vertex_index == -1:
#     print("  Skipping path check due to invalid landmark indices.")
#     path_possible = False
# elif num_components > 1:
#     print("  Skipping path check because mesh has disconnected components.")
#     # More advanced check: find which component each landmark belongs to.
#     # This requires iterating through components and mapping original indices.
#     # For now, we assume if >1 components, path *might* fail.
#     # Let's try it anyway, but warn the user.
#     print("    Warning: Path check might fail or be incorrect due to disconnected components.")
# elif num_components == -1: # Failed split check
#      print("  Skipping path check due to failure in component/manifold check.")
#      path_possible = False
# elif not is_initially_watertight and not mesh.is_watertight : # If split didn't fix watertightness
#      print("  Skipping path check because mesh appears non-watertight/non-manifold.")
#      path_possible = False

# if path_possible:
#     try:
#         path_nodes = trimesh.graph.shortest_path(
#             mesh,
#             start_node=wrist_vertex_index,
#             end_node=shoulder_vertex_index
#         )

#         if path_nodes is not None and len(path_nodes) > 0:
#              vertices = mesh.vertices[path_nodes]
#              distances = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
#              path_distance = np.sum(distances)

#              print(f"  Result: Path found! ✅")
#              print(f"    Distance: {path_distance} (in mesh units, likely meters)")
#              print(f"    Number of vertices in path: {len(path_nodes)}")
#         else:
#             print(f"  Result: No path found between the specified landmarks. ❌")
#             print("          This is the likely reason for the infinite 'arm_length'.")
#             print("          Possible causes: Landmarks on disconnected components, mesh errors.")

#     except Exception as e:
#         print(f"  Error during shortest path calculation: {e} ❌")
#         print("          This could indicate mesh connectivity issues or numerical problems.")

# print("\n--- Checks Complete ---")
