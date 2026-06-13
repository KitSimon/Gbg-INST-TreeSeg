# Gbg-INST-TreeSeg — tree-crown instance segmentation pipeline.
# Copyright (C) 2026 KitSimon
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Interactive AOI picker for FG_2 inference test runs.

Opens a native Qt window with QWebEngineView embedding Leaflet + Leaflet.draw,
shows the ortho's footprint as a polygon over an Esri Imagery / OSM basemap,
and returns the user-drawn rectangle's bounds in the ortho's raster CRS so
FG_2 can clip the ortho + semantic raster to that subset before running
Cellpose.

Used by FG_2_inference_runner.py when TEST_SUBSET_PICKER=True. Lives outside
the inference pipeline itself so the inference path stays untouched — the
runner just swaps in clipped VRTs after pick_aoi() returns.

WSL2 caveat: QWebEngineView under WSLg generally works but depends on a
functional GL stack; if it fails to render, install PyQt5/PySide6's
matching webengine package and ensure WSLg is up-to-date.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Tuple


def _ensure_proj_data_compatible_with_rasterio() -> None:
    """Work around the multi-libproj mismatch this conda env has (see README).

    Several wheels in this env each ship their own libproj + proj.db. The
    versions on disk are not all equal:
        env share/proj                 — v1.2
        pyproj/proj_dir/share/proj     — v1.2
        rasterio/proj_data             — v1.3
        pyogrio/proj_data              — v1.5
    rasterio's libproj is loaded as soon as `import rasterio` runs and
    expects v1.3+; pointing PROJ_DATA at the env-shared v1.2 db (the conda
    default) makes every Transformer.from_crs(...) call fail with 'Error
    creating Transformer from CRS.'

    Fix: point PROJ_DATA at rasterio's own bundled v1.3 proj.db. That works
    for rasterio (its expected version) and for pyproj (pyproj's libproj
    accepts >=v1.2). We only override if the rasterio bundle is present —
    if not, we leave whatever the user has alone.
    """
    try:
        import rasterio as _rio
    except ImportError:
        return
    bundled = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
    if os.path.isfile(os.path.join(bundled, "proj.db")):
        os.environ["PROJ_DATA"] = bundled


_ensure_proj_data_compatible_with_rasterio()

import rasterio  # noqa: E402  (must come after PROJ_DATA tweak)
from pyproj import CRS as _PCRS  # noqa: E402
from pyproj import Transformer  # noqa: E402

# Try PyQt5 first (already installed in this env), fall back to PySide6.
_QT_BINDING = None
try:
    from PyQt5.QtCore import QObject, QUrl, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtWebChannel import QWebChannel
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWidgets import QApplication, QMainWindow
    _QT_BINDING = "PyQt5"
except ImportError:
    try:
        from PySide6.QtCore import QObject, QUrl, Signal, Slot
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication, QMainWindow
        _QT_BINDING = "PySide6"
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# HTML / JS for the picker window
# ---------------------------------------------------------------------------
#
# Loaded into QWebEngineView with setHtml(). Leaflet + Leaflet.draw are pulled
# from unpkg; this requires internet access on the host running the picker.
# The two basemap layers (Esri Imagery, OpenStreetMap) are fetched from their
# public tile servers — no API keys required.
#
# The QWebChannel bridge exposes a single Slot, bridge.submit(bbox_json),
# which the JS calls when the user clicks "Confirm selection".

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>FG_2 AOI picker</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; }
  #map { width:100%; height:100vh; }
  #toolbar { position:absolute; top:8px; left:60px; z-index:1000;
             background:white; padding:6px 10px; border-radius:4px;
             box-shadow:0 1px 4px rgba(0,0,0,.3);
             font:13px sans-serif; }
  #toolbar button { padding:4px 12px; cursor:pointer; margin-left:6px; }
  #status { font-family:monospace; }
</style>
</head>
<body>
<div id="map"></div>
<div id="toolbar">
  <span id="status">Draw a rectangle on the map to select a test subset.</span>
  <button id="submit-btn" disabled>Confirm selection</button>
</div>
<script>
const FOOTPRINT = __FOOTPRINT__;
const FOOTPRINT_BOUNDS = __FOOTPRINT_BOUNDS__;

const map = L.map('map');

const osm = L.tileLayer(
  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  { attribution: '© OpenStreetMap contributors', maxZoom: 19 }
);
const esri = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Tiles © Esri', maxZoom: 19 }
);
esri.addTo(map);

L.control.layers(
  { 'Esri Imagery': esri, 'OpenStreetMap': osm },
  {}
).addTo(map);

const footprint = L.polygon(FOOTPRINT, {
  color: '#ff5500', weight: 2, fillOpacity: 0.05
}).addTo(map);
map.fitBounds(FOOTPRINT_BOUNDS);

const drawnItems = new L.FeatureGroup().addTo(map);
const drawControl = new L.Control.Draw({
  edit: { featureGroup: drawnItems, edit: true, remove: true },
  draw: {
    polygon: false, polyline: false, circle: false,
    marker: false, circlemarker: false,
    rectangle: { shapeOptions: { color: '#0066ff', weight: 2 } }
  }
});
map.addControl(drawControl);

let currentBbox = null;
function setBbox(layer) {
  drawnItems.clearLayers();
  drawnItems.addLayer(layer);
  const b = layer.getBounds();
  currentBbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  document.getElementById('status').textContent =
    'WGS84  W=' + currentBbox[0].toFixed(5) +
    '  S=' + currentBbox[1].toFixed(5) +
    '  E=' + currentBbox[2].toFixed(5) +
    '  N=' + currentBbox[3].toFixed(5);
  document.getElementById('submit-btn').disabled = false;
}
map.on(L.Draw.Event.CREATED, e => setBbox(e.layer));
map.on(L.Draw.Event.EDITED, e => e.layers.eachLayer(l => setBbox(l)));
map.on(L.Draw.Event.DELETED, () => {
  currentBbox = null;
  document.getElementById('submit-btn').disabled = true;
  document.getElementById('status').textContent =
    'Draw a rectangle on the map to select a test subset.';
});

new QWebChannel(qt.webChannelTransport, function (channel) {
  window.bridge = channel.objects.bridge;
});

document.getElementById('submit-btn').addEventListener('click', function () {
  if (currentBbox && window.bridge) {
    window.bridge.submit(JSON.stringify(currentBbox));
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------

def _ortho_pyproj_crs(ortho_path: str, fallback_crs_epsg: int) -> _PCRS:
    """Return a pyproj CRS for the ortho, working around the PROJ db
    version mismatch that bites elsewhere in this pipeline.

    Order of preference:
      1. ortho.crs.to_epsg() — most reliable when set.
      2. ortho.crs.to_wkt() through pyproj — works when the WKT round-trips.
      3. fallback_crs_epsg — last-resort, matches the rest of the codebase.
    """
    with rasterio.open(ortho_path) as ortho:
        ortho_crs = ortho.crs
    if ortho_crs is not None:
        try:
            epsg = ortho_crs.to_epsg()
            if epsg is not None:
                return _PCRS.from_epsg(epsg)
        except Exception:
            pass
        try:
            return _PCRS.from_wkt(ortho_crs.to_wkt())
        except Exception:
            pass
    return _PCRS.from_epsg(fallback_crs_epsg)


def _ortho_footprint_wgs84(
    ortho_path: str, fallback_crs_epsg: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Return (footprint_latlng, bounds_latlng) for Leaflet.

    The ortho bounds are densified along each edge before reprojection so a
    curved-projection footprint isn't underrepresented by just four corners.
    """
    with rasterio.open(ortho_path) as ortho:
        bounds = ortho.bounds

    src_crs = _ortho_pyproj_crs(ortho_path, fallback_crs_epsg)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    n = 32
    xs = [bounds.left + i * (bounds.right - bounds.left) / n for i in range(n + 1)]
    ys = [bounds.bottom + i * (bounds.top - bounds.bottom) / n for i in range(n + 1)]
    pts = []
    for x in xs:
        pts.append((x, bounds.bottom))
    for y in ys:
        pts.append((bounds.right, y))
    for x in reversed(xs):
        pts.append((x, bounds.top))
    for y in reversed(ys):
        pts.append((bounds.left, y))

    lonlat = [transformer.transform(x, y) for x, y in pts]
    latlng = [[lat, lon] for lon, lat in lonlat]
    lats = [p[0] for p in latlng]
    lons = [p[1] for p in latlng]
    return latlng, [[min(lats), min(lons)], [max(lats), max(lons)]]


def _wgs84_bbox_to_ortho_crs(
    bbox_wgs84: Tuple[float, float, float, float],
    ortho_path: str,
    fallback_crs_epsg: int,
) -> Tuple[float, float, float, float]:
    """Transform a (W, S, E, N) WGS84 bbox to an axis-aligned bbox in the
    ortho's raster CRS.

    The four corners of a WGS84 rectangle don't form an axis-aligned rectangle
    in a projected CRS, so we take min/max of all four projected corners.
    Result is slightly larger than the user's lat/lon rectangle, which is
    fine for picking a test region.
    """
    west, south, east, north = bbox_wgs84
    dst_crs = _ortho_pyproj_crs(ortho_path, fallback_crs_epsg)
    rev = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = [], []
    for lon, lat in [(west, south), (east, south), (east, north), (west, north)]:
        x, y = rev.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Picker entry point
# ---------------------------------------------------------------------------

if _QT_BINDING is not None:
    class _Bridge(QObject):
        @Slot(str)
        def submit(self, bbox_json: str) -> None:
            self.bbox = json.loads(bbox_json)
            self.window.close()


def pick_aoi(
    ortho_path: str, fallback_crs_epsg: int = 3007,
) -> Tuple[float, float, float, float]:
    """Open the picker; return (xmin, ymin, xmax, ymax) in the ortho's CRS.

    Raises RuntimeError if Qt isn't installed or if the user closed the
    window without confirming a selection.
    """
    if _QT_BINDING is None:
        raise RuntimeError(
            "AOI picker requires PyQt5 or PySide6. Install one with:\n"
            "  pip install PyQt5 PyQtWebEngine\n"
            "or\n"
            "  pip install PySide6"
        )

    footprint, bounds_latlng = _ortho_footprint_wgs84(ortho_path, fallback_crs_epsg)
    html = (
        _HTML_TEMPLATE
        .replace("__FOOTPRINT__", json.dumps(footprint))
        .replace("__FOOTPRINT_BOUNDS__", json.dumps(bounds_latlng))
    )

    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle(f"FG_2 AOI picker — {os.path.basename(ortho_path)}")
    win.resize(1100, 800)

    view = QWebEngineView(win)
    bridge = _Bridge()
    bridge.bbox = None
    bridge.window = win
    channel = QWebChannel(view.page())
    channel.registerObject("bridge", bridge)
    view.page().setWebChannel(channel)
    view.setHtml(html, QUrl("about:blank"))
    win.setCentralWidget(view)
    win.show()

    app.exec_() if hasattr(app, "exec_") else app.exec()

    if bridge.bbox is None:
        raise RuntimeError(
            "AOI picker closed without a confirmed selection — aborting."
        )

    bbox_proj = _wgs84_bbox_to_ortho_crs(
        tuple(bridge.bbox), ortho_path, fallback_crs_epsg
    )
    return bbox_proj


# ---------------------------------------------------------------------------
# Clipping helper
# ---------------------------------------------------------------------------

def clip_to_bbox_vrt(
    input_path: str, output_vrt: str,
    bbox: Tuple[float, float, float, float],
) -> str:
    """Write a VRT clipped to (xmin, ymin, xmax, ymax) using gdal_translate.

    -projwin takes ulx uly lrx lry (in the input's CRS), so for a north-up
    raster the order is (xmin, ymax, xmax, ymin). VRT output is virtually
    free — no pixel copy.
    """
    xmin, ymin, xmax, ymax = bbox
    out_dir = os.path.dirname(output_vrt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "gdal_translate", "-q", "-of", "VRT",
        "-projwin", str(xmin), str(ymax), str(xmax), str(ymin),
        input_path, output_vrt,
    ]
    subprocess.run(cmd, check=True)
    return output_vrt
