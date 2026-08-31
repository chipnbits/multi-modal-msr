"""Patchwise super-resolution reconstruction over an arbitrary geographic region."""

from magsr.reconstruct.blending import (
    BLENDS,
    BlendKind,
    cosine_2d,
    gaussian_2d,
    linear_2d,
    make_blend_weight,
    ones_2d,
)
from magsr.reconstruct.build import (
    Normalizer,
    PatchPlan,
    assign_patch_blocks,
    build_aux_lr_channels,
    infer_patch,
    load_rect_json,
    plan_lr_patches,
    reconstruct_region,
    reconstruct_region_multichannel,
    write_reconstruction_geotiff,
)
from magsr.reconstruct.viz import ReconItem, plot_multi_block_grid, plot_reconstruction

__all__ = [
    "infer_patch",
    "PatchPlan",
    "plan_lr_patches",
    "load_rect_json",
    "make_blend_weight",
    "BlendKind",
    "BLENDS",
    "linear_2d",
    "cosine_2d",
    "gaussian_2d",
    "ones_2d",
    "write_reconstruction_geotiff",
    "Normalizer",
    "reconstruct_region",
    "reconstruct_region_multichannel",
    "build_aux_lr_channels",
    "assign_patch_blocks",
    "plot_reconstruction",
    "plot_multi_block_grid",
    "ReconItem",
]
