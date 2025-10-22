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

from typing import NamedTuple

import torch
import torch.nn as nn
import xray_gaussian_rasterization_voxelization_cuda as _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.mode,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, 'snapshot_fw.dump')
                print('\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.')
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.mode = raster_settings.mode
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        mode = ctx.mode
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            means3D,
            radii,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            mode,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_opacities,
                    _,  # grad_mu
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, 'snapshot_bw.dump')
                print('\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n')
                raise ex
        else:
            (
                grad_means2D,
                grad_opacities,
                _,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)
        grads = (
            grad_means3D,
            grad_means2D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


# Change according to line 45 gaussian_renderer/__init__.py
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    prefiltered: bool
    mode: int
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
