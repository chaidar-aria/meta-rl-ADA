# data_utils.py (modifikasi/penambahan)

import math
import random
import torch
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from device_config import DEVICE
from osrm_utils import get_osrm_distance_cached


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Hitung jarak great-circle (km) antara dua koordinat (derajat).
    """
    R = 6371.0  # radius bumi dalam km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def filter_flood_polygons_for_user(
    flood_gdf,
    user_lat,
    user_lon,
    max_polygons: int = None,
    radius_km: float = None,
    mode: str = "nearest",
):
    """
    Kembalikan subset flood_gdf untuk dipakai pada pengecekan:
      - max_polygons: jumlah maksimum poligon yang ingin digunakan (None = semua)
      - radius_km: jika diberikan, hanya pilih poligon yang centroid-nya dalam radius ini dari user
      - mode: 'nearest' (urut berdasarkan jarak centroid -> ambil terdekat) atau 'random' (sampling acak)
    Tujuan: mengurangi jumlah poligon yang di-scan untuk meningkatkan performa.
    """
    if flood_gdf is None or flood_gdf.empty:
        return flood_gdf

    # Pastikan CRS WGS84 (lat/lon) sehingga centroid() menghasilkan lon/lat yang benar
    try:
        gdf = flood_gdf.to_crs(epsg=4326)
    except Exception:
        gdf = flood_gdf.copy()

    # Hitung centroid lat/lon untuk setiap poligon (beberapa geometri multi -> centroid)
    centroids = gdf.geometry.centroid
    centroid_coords = [(pt.y, pt.x) for pt in centroids]  # (lat, lon)

    # Hitung jarak ke user
    distances = [
        haversine_km(user_lat, user_lon, lat_c, lon_c)
        for lat_c, lon_c in centroid_coords
    ]

    gdf = gdf.assign(_centroid_dist_km=distances)

    # Jika radius_km diberikan, filter dengan radius dulu
    if radius_km is not None:
        gdf = gdf[gdf["_centroid_dist_km"] <= radius_km]

    if gdf.empty:
        return gdf  # tidak ada poligon dalam radius

    # Jika max_polygons ditentukan, batasi jumlah sesuai mode
    if max_polygons is not None and max_polygons > 0 and len(gdf) > max_polygons:
        if mode == "random":
            sampled_idx = random.sample(list(gdf.index), k=max_polygons)
            gdf = gdf.loc[sampled_idx]
        else:  # default 'nearest'
            gdf = gdf.sort_values("_centroid_dist_km").head(max_polygons)

    # Hapus kolom pembantu sebelum kembalikan (opsional)
    gdf = gdf.drop(columns=[c for c in ["_centroid_dist_km"] if c in gdf.columns])

    return gdf


def load_flood_polygons(path_to_shp):
    """
    Memuat Shapefile yang berisi poligon area banjir ke dalam GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(path_to_shp)
        # Pastikan menggunakan sistem koordinat standar (WGS84) untuk lat/lon
        gdf = gdf.to_crs(epsg=4326)
        print(
            f"✅ Shapefile banjir berhasil dimuat dari: {path_to_shp} ({len(gdf)} poligon)"
        )
        return gdf
    except Exception as e:
        print(f"⚠️ Gagal memuat Shapefile banjir: {e}")
        return None


def is_coord_in_flood_zone(flood_gdf, lat, lon):
    """
    Memeriksa apakah sebuah koordinat (lat, lon) berada di dalam salah satu
    poligon banjir.
    """
    if flood_gdf is None or flood_gdf.empty:
        return False  # Anggap aman jika tidak ada data banjir

    # Buat objek Point dari koordinat. Perhatikan urutan (longitude, latitude)
    point = Point(lon, lat)

    # Cek apakah titik tersebut terkandung dalam salah satu geometri poligon
    return flood_gdf.geometry.contains(point).any()


def get_adaptive_evac_candidates(
    user_lat,
    user_lon,
    all_candidates,
    flood_gdf,
    max_radius_km=5.0,
    max_shp_polygons: int = None,
    shp_filter_mode: str = "nearest",
):
    """
    Secara dinamis mencari kandidat evakuasi yang TIDAK berada di zona banjir
    dalam radius yang meningkat.

    Penambahan:
      - max_shp_polygons: batasi berapa banyak poligon SHP yang digunakan saat pengecekan
      - shp_filter_mode: 'nearest' atau 'random'
    """
    # Jika user ingin membatasi poligon, buat subset yang relevan dulu.
    if flood_gdf is not None and max_shp_polygons is not None:
        # gunakan radius yang lebih besar sedikit untuk mencari poligon relevan (mis. 10 km)
        # agar tidak memotong poligon penting ketika max_radius_km kecil
        flood_subset = filter_flood_polygons_for_user(
            flood_gdf,
            user_lat,
            user_lon,
            max_polygons=max_shp_polygons,
            radius_km=max_radius_km * 2.0 if max_radius_km is not None else None,
            mode=shp_filter_mode,
        )
    else:
        flood_subset = flood_gdf

    radius_steps = [1.0, 2.0, 3.0, 4.0, max_radius_km]

    for radius in radius_steps:
        filtered = []
        for evac_lat, evac_lon in all_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            if not np.isfinite(dist_km) or dist_km > radius:
                continue

            # --- LOGIKA BARU: Cek apakah titik evakuasi ada di zona banjir ---
            is_flooded = is_coord_in_flood_zone(flood_subset, evac_lat, evac_lon)
            if is_flooded:
                continue  # Lewati kandidat ini jika berada di area genangan
            # --- AKHIR LOGIKA BARU ---

            # Tambahkan ke daftar kandidat yang valid
            # Angka 0 di akhir menandakan "tidak banjir" (bahaya=0)
            filtered.append((evac_lat, evac_lon, dist_km, 0))

        if filtered:
            # Urutkan hanya berdasarkan jarak karena semua kandidat sudah aman
            filtered.sort(key=lambda x: x[2])
            coords_only = [(lat, lon) for lat, lon, _, _ in filtered]
            return coords_only, radius

    return [], None
