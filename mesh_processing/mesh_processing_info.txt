Mesh processing stage: there are four different scripts included in this folder. Processing should be applied in the order they are presented here.

>step_0_make_suture_meshes.py: a step to create three-dimensional meshes from 3D .tiff files. A much faster export strategy than manual mesh export from Avizo

>step_0.5_single_invert_check.py: an optional step to change normals orientation in the meshes prior to processing

>step_1_make_meshes_manifold.ply: a step for making meshes manifold and watertight to allow the calculation of complexity metrics

>step_2_rescale_meshes_safely.ply: a step for rescaling meshes according to their correct size. As SPROUT creates agnostic (voxel size 1,1,1) output while keeping original bounding box information, this rescaling step is necessary to reestablish the original size. This step also calculates mesh volume and centroid size.

