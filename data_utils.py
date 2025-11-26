# data_utils.py
import os
import geopandas as gpd
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point
import numpy as np


# --- BAGIAN 1: DATA ELEVASI (BARU) ---
def get_elevation_at_point(tif_path, lat, lon):
    """
    Mengambil nilai elevasi (meter) dari file GeoTIFF pada koordinat (lat, lon).
    """
    try:
        with rasterio.open(tif_path) as src:
            # Cek CRS, lakukan transformasi jika proyeksi file bukan Lat/Lon (WGS84)
            if src.crs.to_string() != "EPSG:4326":
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat

            # Sampling nilai pixel
            try:
                # index() mengembalikan (row, col)
                row, col = src.index(x, y)
                data = src.read(1)

                # Pastikan indeks valid
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    elev = data[row, col]
                    # Filter nilai nodata (biasanya minus besar atau -9999)
                    if elev < -100:
                        return 0.0
                    return float(elev)
                else:
                    return 0.0
            except Exception:
                return 0.0
    except Exception as e:
        # print(f"⚠️ Warning: Gagal baca elevasi: {e}")
        return 0.0


# --- BAGIAN 2: DATA EVAKUASI (GEOJSON/SHP) ---
def load_evac_candidates(path):
    """
    Memuat titik evakuasi dari GeoJSON atau SHP.
    Mengembalikan list of dict: [{'coord': (lat, lon), 'name': 'Nama Tempat'}]
    """
    try:
        gdf = gpd.read_file(path)

        # Pastikan CRS WGS84
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        evac_data = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                # Ambil nama tempat dari kolom 'Evakuasi' (sesuai dataset Anda)
                name = row.get("Evakuasi", f"Titik_{row.name}")

                evac_data.append(
                    {"coord": (geom.y, geom.x), "name": str(name)}  # (lat, lon)
                )

        print(
            f"✅ Berhasil memuat {len(evac_data)} titik evakuasi dari {os.path.basename(path)}"
        )
        return evac_data
    except Exception as e:
        print(f"⚠️ Gagal memuat data evakuasi: {e}")
        return []


# --- BAGIAN 3: DATA BANJIR ---
def load_flood_polygons(path_to_shp):
    """Memuat Shapefile banjir ke GeoDataFrame."""
    try:
        gdf = gpd.read_file(path_to_shp)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        print(f"✅ Berhasil memuat {len(gdf)} poligon banjir.")
        return gdf
    except Exception as e:
        print(f"⚠️ Gagal memuat Shapefile banjir: {e}")
        return None


def is_coord_in_flood_zone(flood_gdf, lat, lon):
    """Cek apakah koordinat ada di dalam zona banjir."""
    if flood_gdf is None or flood_gdf.empty:
        return False
    point = Point(lon, lat)  # Shapely pakai (lon, lat)
    # Gunakan spatial index untuk performa lebih cepat (jika ada)
    # Untuk simplifikasi, kita pakai contains biasa dulu
    return flood_gdf.geometry.contains(point).any()
