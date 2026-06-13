"""
Machine-specific external paths — TEMPLATE.

Copy this file to local_paths.py and replace the placeholder values with the
absolute paths on your machine. local_paths.py is gitignored, so your real
paths never get committed. project_paths.py imports the names below and
re-exports them to the rest of the pipeline; if local_paths.py is missing it
raises a clear error pointing you here.
"""

# Source orthophoto mosaic (GeoTIFF or VRT). RGB-IR or RGB. Used by FG_2,
# FG_2a, and FG_2b.
ORTHO_PATH = "/path/to/ortho/_mosaik.vrt"

# Root of the Gbg-SEM-TreeSeg working copy (stage-1 semantic segmentation).
# FG_0 reads the AF_0 crown polygons from it; FG_2 reads the AF_3 semantic raster.
GBG_SEM_ROOT = "/path/to/Gbg-SEM-TreeSeg/AerialFormer"

# Directory of LAS tiles (one per ortho tile). Used by FG_3.
LAS_DIR = "/path/to/las_tiles"

# Directory of ortho GeoTIFF tiles. Each LAS tile is paired with the TIF of the
# same stem; the TIF grid defines the per-tile DTM rasterization. Used by FG_3.
TIF_DIR = "/path/to/ortho_tiles"
