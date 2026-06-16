import os 
import sys
import trimesh
import pymeshlab
import csv
import numpy as np

input_dir = os.path.abspath("C:/Users/goswamilab/Downloads/Centroscymnus_coelolepis_denticles_STL/processed/plyd/") #directory where your unprocessed meshes are located
output_dir = os.path.abspath("C:/Users/goswamilab/Downloads/Centroscymnus_coelolepis_denticles_STL/processed/") #output directory

if input_dir == output_dir:
    print("Error: input_dir and output_dir are the same. You will overwrite your lovely scans! Choose again.")
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ms = pymeshlab.MeshSet()

csv_path = os.path.join(output_dir, "mesh_volume_centroid_Centroscymnus.csv")
processed = set()

# Load previously processed filenames
if os.path.exists(csv_path):
    with open(csv_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)  # Skip header
        for row in reader:
            if row:
                processed.add(row[0])

# Open CSV for appending new entries
with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.ply') or filename in processed:
            continue

        print(f"🔹 Processing {filename}")
        input_path = os.path.join(input_dir, filename)

        try:
            mesh = trimesh.load_mesh(input_path, process=False)
            if mesh.is_empty:
                print(f"⚠️ Empty mesh: {filename}, skipping")
                continue

            components = mesh.split(only_watertight=False)
            largest_comp = max(components, key=lambda m: m.faces.shape[0])
            print(f"   ➤ Largest component has {largest_comp.faces.shape[0]} faces")

            temp_path = os.path.join(output_dir, "__temp.ply")
            largest_comp.export(temp_path)

            ms.load_new_mesh(temp_path)
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes(maxholesize=50, selected=False, newfaceselected=False)
            #ms.meshing_invert_face_orientation()

            ms.generate_surface_reconstruction_screened_poisson(
                depth=8,
                fulldepth=5,
                cgdepth=0,
                scale=1.1,
                samplespernode=1.5,
                pointweight=4.0,
                iters=8,
                preclean=False,
                confidence=False
            )

            output_path = os.path.join(output_dir, filename)
            ms.save_current_mesh(output_path)
            ms.clear()

            cleaned_mesh = trimesh.load_mesh(output_path, process=False)

            vol = cleaned_mesh.volume if cleaned_mesh.is_watertight else 0.0
            verts = cleaned_mesh.vertices
            centroid = verts.mean(axis=0)
            centroid_size = np.sqrt(np.sum(np.linalg.norm(verts - centroid, axis=1)**2))

            print(f"   ✅ Volume: {vol:.6f}, Centroid Size: {centroid_size:.6f}")

            writer.writerow([filename, vol, centroid_size])
            csvfile.flush()  # Ensure write is saved after each mesh

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
