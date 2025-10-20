# data_utils.py
# Berisi fungsi untuk memuat data banjir (Shapefile) dan memfilter kandidat evakuasi.

import torch
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from device_config import DEVICE
from osrm_utils import get_osrm_distance_cached


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
    user_lat, user_lon, all_candidates, flood_gdf, max_radius_km=5.0
):
    """
    Secara dinamis mencari kandidat evakuasi yang TIDAK berada di zona banjir
    dalam radius yang meningkat.
    """
    radius_steps = [1.0, 2.0, 3.0, 4.0, max_radius_km]

    for radius in radius_steps:
        filtered = []
        for evac_lat, evac_lon in all_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            if not np.isfinite(dist_km) or dist_km > radius:
                continue

            # --- LOGIKA BARU: Cek apakah titik evakuasi ada di zona banjir ---
            is_flooded = is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon)
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
