#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizer,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianRasterizationSettings,
)

from r2_gaussian.arguments import PipelineParams
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.gaussian.gaussian_model import GaussianModel

MAX_N_VOXELS = 256


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
    """
    if nVoxel[0] > MAX_N_VOXELS or nVoxel[1] > MAX_N_VOXELS or nVoxel[2] > MAX_N_VOXELS:
        # Split the volume query into smaller sub-queries
        center = torch.tensor(center, dtype=torch.float32)
        sVoxel = torch.tensor(sVoxel, dtype=torch.float32)
        nVoxel = torch.tensor(nVoxel, dtype=torch.int32)
        bbox_min = center - 0.5 * sVoxel
        bbox_max = center + 0.5 * sVoxel

        n_tile_z = (nVoxel[2] + MAX_N_VOXELS - 1) // MAX_N_VOXELS
        n_tile_y = (nVoxel[1] + MAX_N_VOXELS - 1) // MAX_N_VOXELS
        n_tile_x = (nVoxel[0] + MAX_N_VOXELS - 1) // MAX_N_VOXELS

        vol = torch.zeros(
            (nVoxel[0], nVoxel[1], nVoxel[2]),
            dtype=torch.float32,
            device=pc.get_xyz.device,
        )
        radii = []
        for tz in range(n_tile_z):
            for ty in range(n_tile_y):
                for tx in range(n_tile_x):
                    sub_nVoxel = torch.zeros(3, dtype=torch.int32)
                    sub_sVoxel = torch.zeros(3, dtype=torch.float32)
                    sub_center = torch.zeros(3, dtype=torch.float32)
                    for dim, n_tile, t in zip([0, 1, 2], [n_tile_x, n_tile_y, n_tile_z], [tx, ty, tz]):
                        start_idx = t * MAX_N_VOXELS
                        end_idx = min((t + 1) * MAX_N_VOXELS, int(nVoxel[dim]))
                        sub_nVoxel[dim] = end_idx - start_idx
                        sub_sVoxel[dim] = sVoxel[dim] * float(sub_nVoxel[dim]) / float(nVoxel[dim])
                        sub_center[dim] = bbox_min[dim] + (
                            (start_idx + 0.5 * sub_nVoxel[dim]) * sVoxel[dim] / float(nVoxel[dim])
                        )

                    sub_result = query(
                        pc=pc,
                        center=sub_center.tolist(),
                        nVoxel=sub_nVoxel.tolist(),
                        sVoxel=sub_sVoxel.tolist(),
                        pipe=pipe,
                        scaling_modifier=scaling_modifier,
                    )

                    vol[
                        tx * MAX_N_VOXELS : tx * MAX_N_VOXELS + sub_nVoxel[0],
                        ty * MAX_N_VOXELS : ty * MAX_N_VOXELS + sub_nVoxel[1],
                        tz * MAX_N_VOXELS : tz * MAX_N_VOXELS + sub_nVoxel[2],
                    ] = sub_result['vol']
                    radii.append(torch.stack(sub_result['radii'], dim=0))

        # Combine results
        radii = torch.stack(radii, dim=0).max(dim=0)[0]
        radii = (radii[0], radii[1], radii[2])

        return {
            'vol': vol,
            'radii': radii,
        }

    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        'vol': vol_pred,
        'radii': radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError('Unsupported mode!')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        'render': rendered_image,
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
    }
