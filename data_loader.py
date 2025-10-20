# data_loader.py
# Berisi fungsi untuk memuat data dari berbagai sumber, termasuk Shapefile.
# Diperbarui untuk menangani Shapefile tanpa informasi CRS.

import os
import geopandas as gpd
import json


def load_evac_candidates_shp(shp_path, target_crs=None):
    """
    Memuat titik evakuasi dari Shapefile.

    PERBAIKAN: Fungsi ini sekarang menangani file yang tidak memiliki CRS.
    """
    try:
        gdf = gpd.read_file(shp_path)

        # === BLOK PERBAIKAN CRS ===
        # 1. Cek jika CRS tidak ada (geometri "naif")
        if gdf.crs is None:
            print(
                f"⚠️ Peringatan: CRS tidak ditemukan di {os.path.basename(shp_path)}. Mengasumsikan WGS84 (EPSG:4326)."
            )
            # 2. Tetapkan CRS default (WGS84)
            gdf.set_crs("EPSG:4326", inplace=True)

        # 3. Jika CRS target diberikan (dari file banjir), samakan CRS-nya.
        if target_crs and gdf.crs != target_crs:
            print(
                f"Menyamakan CRS dari {os.path.basename(shp_path)} ke {target_crs.name}..."
            )
            gdf = gdf.to_crs(target_crs)
        # === AKHIR BLOK PERBAIKAN ===

        evac_points = []
        if "geometry" not in gdf.columns:
            raise ValueError("Kolom 'geometry' tidak ditemukan di Shapefile.")

        for point in gdf.geometry:
            if point is not None and not point.is_empty:
                # Koordinat diekstrak sebagai (latitude, longitude)
                evac_points.append((point.y, point.x))

        print(
            f"✅ {len(evac_points)} kandidat evakuasi berhasil dimuat dari: {os.path.basename(shp_path)}"
        )
        return evac_points

    except Exception as e:
        print(f"⚠️ Gagal memuat Shapefile kandidat evakuasi: {e}")
        return []


def generate_user_coords_from_shp(shp_path, n=100):
    """
    Menghasilkan titik-titik pengguna acak di dalam area poligon dari Shapefile.
    """
    try:
        gdf = gpd.read_file(shp_path)
        # Ambil batas total dari semua poligon
        total_bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = total_bounds

        user_points = []
        while len(user_points) < n:
            # Hasilkan titik acak di dalam kotak batas
            import random

            rand_x = random.uniform(xmin, xmax)
            rand_y = random.uniform(ymin, ymax)
            point = gpd.points_from_xy([rand_x], [rand_y], crs=gdf.crs)[0]

            # Pastikan titik berada di dalam salah satu poligon
            if gdf.contains(point).any():
                user_points.append((point.y, point.x))  # Simpan sebagai (lat, lon)

        print(f"✅ {len(user_points)} koordinat pengguna berhasil dibuat dari SHP.")
        return user_points
    except Exception as e:
        print(f"⚠️ Gagal menghasilkan koordinat pengguna dari SHP: {e}")
        return []


def load_user_coords(json_path):
    """Memuat koordinat pengguna dari file JSON."""
    try:
        with open(json_path, "r") as f:
            coords = json.load(f)
        print(f"✅ {len(coords)} koordinat pengguna dimuat dari JSON.")
        return coords
    except Exception as e:
        print(f"⚠️ Gagal memuat koordinat pengguna dari JSON: {e}")
        return []


def generate_and_save_user_coords_if_needed(shp_path, json_path, n=100):
    """
    Hanya menghasilkan dan menyimpan koordinat jika file JSON belum ada.
    """
    if os.path.exists(json_path):
        print(f"File koordinat pengguna '{json_path}' sudah ada. Melewatkan pembuatan.")
        return

    print(f"Menghasilkan {n} koordinat pengguna dari '{shp_path}'...")
    coords = generate_user_coords_from_shp(shp_path, n=n)
    if coords:
        with open(json_path, "w") as f:
            json.dump(coords, f, indent=2)
        print(f"Koordinat pengguna disimpan ke '{json_path}'.")
