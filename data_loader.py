# # data_loader.py
# # ---------------------------
# # Loader untuk user_coords, evacuation candidates, dan MMI dari file JSON

# import json
# import os
# from typing import List, Tuple
# from shapely.geometry import Point
# import geopandas as gpd
# import numpy as np


# def generate_and_save_user_coords_if_needed(
#     shp_path: str, json_path: str, n: int = 1000
# ):
#     if os.path.exists(json_path):
#         print(f"✅ File user JSON sudah ada: {json_path}")
#         return

#     print(f"⚠️ File user JSON belum ada, generate dari SHP...")

#     gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
#     polygon = gdf.geometry.union_all()
#     minx, miny, maxx, maxy = polygon.bounds

#     points = []
#     attempts = 0
#     while len(points) < n and attempts < n * 100:
#         lat = np.random.uniform(miny, maxy)
#         lon = np.random.uniform(minx, maxx)
#         if polygon.contains(Point(lon, lat)):
#             points.append({"latitude": lat, "longitude": lon})
#         attempts += 1

#     with open(json_path, "w") as f:
#         json.dump(points, f, indent=2)

#     print(f"✅ Titik user disimpan ke: {json_path}")


# def generate_user_coords_from_shp(shp_path: str, n: int = 250):
#     gdf = gpd.read_file(shp_path)
#     gdf = gdf.to_crs("EPSG:4326")
#     polygon = gdf.geometry.union_all()

#     points = []
#     minx, miny, maxx, maxy = polygon.bounds
#     attempts = 0
#     while len(points) < n and attempts < n * 50:
#         lat = np.random.uniform(miny, maxy)
#         lon = np.random.uniform(minx, maxx)
#         p = Point(lon, lat)
#         if polygon.contains(p):
#             points.append((lat, lon))
#         attempts += 1

#     if len(points) < n:
#         print(f"⚠️ Hanya dapat menghasilkan {len(points)} dari {n} titik")
#     return points


# # ===== Load User Coordinates dari JSON =====
# def load_user_coords(path: str) -> List[Tuple[float, float]]:
#     with open(path, "r") as f:
#         data = json.load(f)
#     coords = [(entry["latitude"], entry["longitude"]) for entry in data]
#     return coords


# # ===== Load Dataset Event (Evacuation + MMI) =====
# def load_event_dataset(event_id: str, dataset_dir: str):
#     path = os.path.join(dataset_dir, f"{event_id}.json")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset untuk event {event_id} tidak ditemukan.")

#     with open(path, "r") as f:
#         data = json.load(f)

#     # Ambil titik evakuasi sebagai kandidat
#     evac_points = [
#         (d["latitude"], d["longitude"]) for d in data.get("evacuationData", [])
#     ]

#     # Format ulang MMI untuk fungsi RL
#     mmi_points = []
#     for m in data.get("mmiData", []):
#         mmi_points.append(
#             {"lat": m["latitude"], "lon": m["longitude"], "mmi": m["mmi_level"]}
#         )

#     return evac_points, mmi_points


# data_loader.py
# ---------------------------
# Loader untuk user_coords, evacuation candidates (GeoJSON), dan MMI (cont_mmi JSON)

import json
import os
from typing import List, Tuple
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from typing import List, Tuple, Iterable, Union

def generate_and_save_user_coords_if_needed(
    shp_path: str, json_path: str, n: int = 1000
):
    if os.path.exists(json_path):
        print(f"✅ File user JSON sudah ada: {json_path}")
        return

    print(f"⚠️ File user JSON belum ada, generate dari SHP...")

    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    polygon = gdf.geometry.union_all()
    minx, miny, maxx, maxy = polygon.bounds

    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        lat = np.random.uniform(miny, maxy)
        lon = np.random.uniform(minx, maxx)
        if polygon.contains(Point(lon, lat)):
            points.append({"latitude": lat, "longitude": lon})
        attempts += 1

    with open(json_path, "w") as f:
        json.dump(points, f, indent=2)

    print(f"✅ Titik user disimpan ke: {json_path}")


def generate_user_coords_from_shp(shp_path: str, n: int = 250):
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    polygon = gdf.geometry.union_all()

    points = []
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    while len(points) < n and attempts < n * 50:
        lat = np.random.uniform(miny, maxy)
        lon = np.random.uniform(minx, maxx)
        p = Point(lon, lat)
        if polygon.contains(p):
            points.append((lat, lon))
        attempts += 1

    if len(points) < n:
        print(f"⚠️ Hanya dapat menghasilkan {len(points)} dari {n} titik")
    return points


# ===== Load User Coordinates dari JSON =====
def load_user_coords(path: str) -> List[Tuple[float, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    # format: [{"latitude": ..., "longitude": ...}, ...]
    coords = [(entry["latitude"], entry["longitude"]) for entry in data]
    return coords


# # ===== Baru: Load Evacuation Candidates dari GeoJSON =====
# def load_evac_candidates_geojson(geojson_path: str) -> List[Tuple[float, float]]:
#     """
#     Ambil kandidat evakuasi dari FeatureCollection.
#     - Point/MultiPoint → ambil titik langsung
#     - LineString/MultiLineString → ambil semua vertex sebagai kandidat konservatif
#     - Polygon/MultiPolygon → ambil ring luar (outer) sebagai kandidat
#     Output: list[(lat, lon)]
#     """
#     with open(geojson_path, "r", encoding="utf-8") as f:
#         g = json.load(f)

#     feats = g.get("features", [])
#     out: List[Tuple[float, float]] = []

#     def _xy_to_latlon(xy):
#         # GeoJSON CRS84: [lon, lat] → (lat, lon)
#         lon, lat = float(xy[0]), float(xy[1])
#         return (lat, lon)

#     for ft in feats:
#         geom = (ft or {}).get("geometry") or {}
#         gtype = geom.get("type")
#         coords = geom.get("coordinates")
#         if not gtype or coords is None:
#             continue

#         if gtype == "Point":
#             out.append(_xy_to_latlon(coords))

#         elif gtype == "MultiPoint":
#             for xy in coords:
#                 out.append(_xy_to_latlon(xy))

#         elif gtype in ("LineString", "MultiLineString"):
#             seqs = [coords] if gtype == "LineString" else coords
#             for line in seqs:
#                 for xy in line:
#                     out.append(_xy_to_latlon(xy))

#         elif gtype in ("Polygon", "MultiPolygon"):
#             polys = [coords] if gtype == "Polygon" else coords
#             for poly in polys:
#                 if not poly:
#                     continue
#                 outer = poly[0]
#                 for xy in outer:
#                     out.append(_xy_to_latlon(xy))

#     # de-dupe sederhana
#     out = list(dict.fromkeys(out))
#     return out


# ===== Robust Loader: Evacuation Candidates dari GeoJSON =====
def load_evac_candidates_geojson(
    geojson_obj: Union[str, dict],
) -> List[Tuple[float, float]]:
    """
    Ekstrak kandidat titik evakuasi dari GeoJSON:
    - Menerima path (str) atau dict yang sudah ter-parse.
    - Support: FeatureCollection / Feature / Geometry / GeometryCollection.
    - Point/MultiPoint  -> ambil titik langsung.
    - LineString/MultiLineString -> ambil SEMUA vertex (konservatif).
    - Polygon/MultiPolygon -> ambil ring luar (outer) saja.
    - Koordinat GeoJSON diasumsikan CRS84: [lon, lat, (opsional z)] -> (lat, lon).
    - Dedup dengan toleransi pembulatan kecil agar tidak dobel.
    Output: list[(lat, lon)]
    """

    def _xy_to_latlon(xy) -> Tuple[float, float]:
        # GeoJSON: [lon, lat, (z?)] -> (lat, lon)
        lon, lat = float(xy[0]), float(xy[1])
        return (lat, lon)

    def _iter_coords_from_geom(geom: dict) -> Iterable[Tuple[float, float]]:
        if not isinstance(geom, dict):
            return
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Point":
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                yield _xy_to_latlon(coords)

        elif gtype == "MultiPoint":
            for xy in coords or []:
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    yield _xy_to_latlon(xy)

        elif gtype == "LineString":
            for xy in coords or []:
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    yield _xy_to_latlon(xy)

        elif gtype == "MultiLineString":
            for line in coords or []:
                for xy in line or []:
                    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                        yield _xy_to_latlon(xy)

        elif gtype == "Polygon":
            # coords: [outer, hole1, ...]; ambil outer saja
            if coords and isinstance(coords, list) and coords[0]:
                for xy in coords[0]:
                    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                        yield _xy_to_latlon(xy)

        elif gtype == "MultiPolygon":
            for poly in coords or []:
                if poly and isinstance(poly, list) and poly[0]:
                    for xy in poly[0]:
                        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                            yield _xy_to_latlon(xy)

        elif gtype == "GeometryCollection":
            for sub in geom.get("geometries") or []:
                yield from _iter_coords_from_geom(sub or {})

        else:
            # Tipe tak dikenal -> abaikan
            return

    # Boleh path (string) atau dict
    if isinstance(geojson_obj, str):
        with open(geojson_obj, "r", encoding="utf-8") as f:
            g = json.load(f)
    else:
        g = geojson_obj

    pts: List[Tuple[float, float]] = []
    gtype = g.get("type")

    if gtype == "FeatureCollection":
        for ft in g.get("features", []):
            geom = (ft or {}).get("geometry") or {}
            pts.extend(_iter_coords_from_geom(geom))
    elif gtype == "Feature":
        geom = g.get("geometry") or {}
        pts.extend(_iter_coords_from_geom(geom))
    else:
        # Asumsikan ini langsung Geometry
        pts.extend(_iter_coords_from_geom(g))

    # De-duplicate sederhana dengan toleransi pembulatan (~1e-7 derajat)
    seen = set()
    out: List[Tuple[float, float]] = []
    for lat, lon in pts:
        key = (round(lat, 7), round(lon, 7))
        if key in seen:
            continue
        seen.add(key)
        out.append((lat, lon))
    return out


# ===== Baru: Load MMI dari berkas cont_mmi (ShakeMap) =====
def load_mmi_from_cont_mmi(mmi_json_path: str):
    """
    Ekstrak ke list dict [{"lat": ..., "lon": ..., "mmi": ...}, ...]
    - Ambil setiap vertex dari LineString/MultiLineString.
    - Gunakan properties.value sebagai nilai MMI untuk semua vertex di feature tsb.
    """
    with open(mmi_json_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    feats = g.get("features", [])
    out = []

    def _xy_to_latlon(xy):
        lon, lat = float(xy[0]), float(xy[1])
        return (lat, lon)

    for ft in feats:
        props = (ft or {}).get("properties") or {}
        val = props.get("value", None)
        geom = (ft or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if val is None or not gtype or coords is None:
            continue

        # Normalisasi menjadi list of LineString
        if gtype == "LineString":
            lines = [coords]
        elif gtype == "MultiLineString":
            lines = coords
        elif gtype == "Point":
            lines = [[coords]]
        elif gtype == "MultiPoint":
            lines = [coords]
        else:
            # fallback: coba flatten dangkal
            lines = [coords] if isinstance(coords, list) else []

        for line in lines:
            for xy in line:
                lat, lon = _xy_to_latlon(xy)
                out.append({"lat": lat, "lon": lon, "mmi": float(val)})

    return out


# ===== (Opsional) Loader lama gabungan masih bisa dipakai jika diperlukan =====
def load_event_dataset(event_id: str, dataset_dir: str):
    """
    Legacy: membaca 1 file gabungan {event_id}.json yang berisi evacuationData & mmiData.
    Tetap disediakan bila masih dipakai di tempat lain.
    """
    path = os.path.join(dataset_dir, f"{event_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset untuk event {event_id} tidak ditemukan.")

    with open(path, "r") as f:
        data = json.load(f)

    evac_points = [
        (d["latitude"], d["longitude"]) for d in data.get("evacuationData", [])
    ]

    mmi_points = []
    for m in data.get("mmiData", []):
        mmi_points.append(
            {"lat": m["latitude"], "lon": m["longitude"], "mmi": m["mmi_level"]}
        )

    return evac_points, mmi_points
