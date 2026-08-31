"""Dataset primitives and backends.

Two builder functions are the primary load API: `build_wa_datasets` and
`build_ksa_aligned_datasets`. Each reads its defaults from
`configs/datasets.yaml` unless a `cfg=` is passed. Patch-build helpers +
`PatchGridSpec`/`PatchIndex` are exposed for the build scripts under
`scripts/build_wa_dataset/` and `scripts/build_ksa_dataset/`.
"""

from magsr.datasets.augment import random_d4
from magsr.datasets.collate import pool_collate, worker_init_fn
from magsr.datasets.io import clear_source_cache
from magsr.datasets.ksa_shield_aligned import (
    KSAAlignedConfig,
    KSAShieldAlignedDataset,
    build_aligned_block_patches,
    build_ksa_aligned_datasets,
    iter_aligned_blocks,
)
from magsr.datasets.patching import (
    CellGridSpec,
    PatchGridSpec,
    PatchIndex,
    PatchWindow,
    assign_cell_splits,
    bucket_patches_by_cell,
    save_mask_geotiff,
    sliding_window_patches,
)
from magsr.datasets.western_australia import (
    WAConfig,
    WAGoldfieldsPatchDataset,
    build_wa_datasets,
    compute_global_percentiles,
    generate_wa_patch_pairs,
)

__all__ = [
    # Patch primitives (scripts + tests)
    "PatchGridSpec",
    "PatchWindow",
    "PatchIndex",
    "sliding_window_patches",
    "save_mask_geotiff",
    "clear_source_cache",
    # Buffered cell-grid splitting (Senyard et al. 2026 protocol)
    "CellGridSpec",
    "bucket_patches_by_cell",
    "assign_cell_splits",
    # WA
    "WAConfig",
    "compute_global_percentiles",
    "generate_wa_patch_pairs",
    "WAGoldfieldsPatchDataset",
    "build_wa_datasets",
    # KSA aligned
    "KSAAlignedConfig",
    "KSAShieldAlignedDataset",
    "build_aligned_block_patches",
    "iter_aligned_blocks",
    "build_ksa_aligned_datasets",
    # Torch helpers
    "pool_collate",
    "worker_init_fn",
    "random_d4",
]
