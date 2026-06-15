import os
import sys
import trimesh
import pymeshlab
import csv
import numpy as np

input_dir = os.path.abspath("E:/Marco_backups/suture_gen/matched_rescaled")
output_dir = os.path.abspath("E:/Marco_backups/suture_gen/matched_rescaled/single_element")

if input_dir == output_dir:
    print("Error: input_dir and output_dir are the same.")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

ms = pymeshlab.MeshSet()

csv_path = os.path.join(output_dir, "mesh_vol_centroid.csv")
processed = set()

# Load processed files
if os.path.exists(csv_path):
    with open(csv_path, mode='r') as f:
        for row in csv.reader(f):
            if row:
                processed.add(row[0])

with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".ply") or filename in processed:
            continue

        print(f"\n🔹 Processing {filename}")
        input_path = os.path.join(input_dir, filename)

        try:
            mesh = trimesh.load_mesh(input_path, process=False)

            if mesh.is_empty:
                print("⚠️ Empty mesh, skipping")
                continue

            # --- Largest component ---
            components = mesh.split(only_watertight=False)
            largest = max(components, key=lambda m: m.faces.shape[0])
            print(f"   ➤ Faces: {largest.faces.shape[0]}")

            # --- TEMP FILE (overwrite each time) ---
            temp_path = os.path.join(output_dir, "__temp.ply")
            largest.export(temp_path)

            # --- Load into PyMeshLab ---
            ms.load_new_mesh(temp_path)

            # Save once to normalize mesh state
            ms.save_current_mesh(temp_path)
            ms.clear()

            # --- Measure initial volume ---
            test_mesh = trimesh.load_mesh(temp_path, process=False)
            test_mesh.fix_normals()
            initial_vol = test_mesh.volume

            print(f"   ➤ Initial volume: {initial_vol:.6f}")

            # --- Fix orientation if needed ---
            if initial_vol < 0:
                print("   ➤ Inverting normals")

                ms.load_new_mesh(temp_path)
                ms.meshing_invert_face_orientation()
                ms.save_current_mesh(temp_path)
                ms.clear()

                # reload after inversion
                test_mesh = trimesh.load_mesh(temp_path, process=False)

            # --- FINAL SAVE ONLY ---
            output_path = os.path.join(output_dir, filename)
            test_mesh.export(output_path)

            final_vol = test_mesh.volume if test_mesh.is_watertight else 0.0

            verts = test_mesh.vertices
            centroid = verts.mean(axis=0)
            centroid_size = np.sqrt(np.sum(np.linalg.norm(verts - centroid, axis=1)**2))

            print(f"   ✅ Final volume: {final_vol:.6f}")
            print(f"   ✅ Centroid size: {centroid_size:.6f}")

            writer.writerow([filename, final_vol, centroid_size])
            csvfile.flush()

        except Exception as e:
            print(f"❌ Error: {e}")
            continue
