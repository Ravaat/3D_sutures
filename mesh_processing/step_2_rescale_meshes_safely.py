import os
import sys
import trimesh
import csv
import numpy as np
import pandas as pd

# Directories
#input_dir = os.path.abspath("D:/Marco_backups/suture_gen/batch_4_cleaned")
input_dir = os.path.abspath("E:/Marco_backups/suture_gen/matched")
output_dir = os.path.abspath("E:/Marco_backups/suture_gen/matched_rescaled")

if input_dir == output_dir:
    print("Error: input_dir and output_dir are the same. You will overwrite your scans! Choose again.")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

# Load voxel size CSV
# expected columns: species, x, y, z   (or change names below if different)
voxel_csv_path = "E:/Marco_backups/pipeline/cleaned_twice/voxres.csv"
voxel_df = pd.read_csv(voxel_csv_path)

def get_voxel_sizes_for_species(species_name):
    row = voxel_df[voxel_df["species"] == species_name]
    if row.empty:
        print(f"⚠️ Warning: species '{species_name}' not found in voxel size CSV. Using (1,1,1)")
        return (1.0, 1.0, 1.0)

    # make sure these match your CSV column names
    return tuple(row.iloc[0][["x", "y", "z"]])

# CSV output
csv_path = os.path.join(output_dir, "mesh_rescaled_volume_centroid_fixed_batch_4.csv")

with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "species", "volume_mm3", "centroid_size_mm"])

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".ply"):
            continue

        print(f"Processing {filename}")

        input_path = os.path.join(input_dir, filename)

        parts = filename.split("_")
        if len(parts) < 2:
            print(f"⚠️ Warning: filename '{filename}' does not contain expected species info, skipping")
            continue

        species = f"{parts[0]}_{parts[1]}"

        # safer loader (avoids Scene issues)
        mesh = trimesh.load(input_path, force="mesh", process=False)

        if mesh.is_empty:
            print(f"⚠️ Empty mesh: {filename}, skipping")
            continue

        vx, vy, vz = get_voxel_sizes_for_species(species)

        # ------------------------------------------------------
        # IMPORTANT:
        # marching_cubes returns vertices in (z, y, x) order
        # but your CSV is (x, y, z)
        #
        # so the scale must be (vz, vy, vx)
        # ------------------------------------------------------

        S = np.eye(4)
        S[0, 0] = vz   # mesh axis 0 == z
        S[1, 1] = vy   # mesh axis 1 == y
        S[2, 2] = vx   # mesh axis 2 == x

        mesh.apply_transform(S)

        # Volume (mm^3 if voxel sizes are in mm)
        volume = mesh.volume if mesh.is_watertight else 0.0

        # Centroid size
        verts = mesh.vertices
        centroid = verts.mean(axis=0)
        centroid_size = np.sqrt(
            np.sum(np.linalg.norm(verts - centroid, axis=1) ** 2)
        )

        print(f"Volume: {volume:.6f} mm³, Centroid Size: {centroid_size:.6f} mm")

        output_path = os.path.join(output_dir, filename)
        mesh.export(output_path)

        writer.writerow([filename, species, volume, centroid_size])

        print(f"Saved rescaled mesh: {output_path}")
        
