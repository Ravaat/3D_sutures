import os
import numpy as np
from skimage import measure, io
import trimesh
import tifffile
from concurrent.futures import ProcessPoolExecutor

# Set the base directory where your voxel and pixel data are located
base_dir = r"D:/Marco_backups/suture_gen/others"

# Base output directory
output_base_dir = r"D:/Marco_backups/suture_gen/others"

# Ensure the output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

def process_image(image_path, parent_folder_name):
    """
    Processes a single image file to generate meshes for each class and save them to disk.
    """
    try:
        # Load the segmentation image
        image = io.imread(image_path)
        
        # Find the corresponding voxel size file
        voxel_filename = os.path.basename(image_path).replace("_pixel", "_voxel")
        voxel_path = os.path.join(os.path.dirname(image_path), voxel_filename)
        
        # Open the voxel size file and extract voxel size
        with tifffile.TiffFile(voxel_path) as tif:
            metadata = tif.pages[0].tags
            x_resolution = metadata['XResolution'].value
            voxel_size = x_resolution[1] / x_resolution[0]
        
        # Identify unique classes in the image (excluding background class 0)
        unique_classes = np.unique(image)
        unique_classes = unique_classes[unique_classes != 0]
        
        # Create the output directory for the current parent folder's "plys" directory
        parent_folder_ply_dir = os.path.join(output_base_dir, parent_folder_name, "plys")
        os.makedirs(parent_folder_ply_dir, exist_ok=True)
        
        # Generate and save a mesh for each class
        for cls in unique_classes:
            # Isolate the current class
            foreground = (image == cls)
            foreground = np.fliplr(foreground)  # Optional flipping if required
            
            # Apply the Marching Cubes algorithm to extract the mesh
            vertices, faces, _, _ = measure.marching_cubes(foreground, level=0, spacing=(voxel_size, voxel_size, voxel_size))
            
            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Fix the normals
            mesh.fix_normals()

            # Remove invalid or degenerated faces
            mesh.remove_degenerate_faces()
            
            # Remove isolated vertices
            mesh.remove_unreferenced_vertices()
            
            # Ensure mesh has no NaN coordinates
            valid_vertices = ~np.isnan(mesh.vertices).any(axis=1)
            if not valid_vertices.all():
                mesh.update_vertices(valid_vertices)

            # Save the cleaned mesh to the ply directory with a filename indicating the class and parent folder name
            mesh_filename = f"{parent_folder_name}_{os.path.splitext(os.path.basename(image_path))[0]}_Class_{cls}.ply"
            mesh_path = os.path.join(parent_folder_ply_dir, mesh_filename)
            mesh.export(mesh_path, file_type="ply")
            
        print(f"Successfully processed {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def traverse_and_process_images(base_dir):
    """
    Traverse all subdirectories and process all TIFF files.
    """
    for root, dirs, files in os.walk(base_dir):
        # Get the name of the current parent directory (excluding ".reordered")
        parent_folder_name = os.path.basename(root).replace(".reordered", "")
        
        # List all image files (both .tif and .tiff)
        image_files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff')]
        
        # Process each image file in the current directory
        for image_file in image_files:
            image_path = os.path.join(root, image_file)
            process_image(image_path, parent_folder_name)

# Main function to start the processing
if __name__ == '__main__':
    traverse_and_process_images(base_dir)
