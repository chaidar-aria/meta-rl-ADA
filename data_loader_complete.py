# data_loader_complete.py
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point
import os


# --- 1. Fungsi Generate User dari Area Pemukiman ---
def generate_users_from_settlement(shp_path, n_users=50):
    """
    Mengambil n_users titik acak YANG BERADA DI DALAM poligon pemukiman.
    """
    print(f"📂 Memuat area pemukiman dari: {shp_path}")
    try:
        gdf = gpd.read_file(shp_path)
        # Pastikan CRS adalah WGS84 (Lat/Lon)
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        user_points = []
        # Ambil bounds total untuk mempersempit area random
        minx, miny, maxx, maxy = gdf.total_bounds

        while len(user_points) < n_users:
            # Generate titik random
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

            # Cek apakah titik ini ada di dalam SALAH SATU poligon pemukiman
            # (Gunakan spatial index agar cepat)
            possible_matches_index = list(gdf.sindex.query(pnt))
            possible_matches = gdf.iloc[possible_matches_index]
            if possible_matches.contains(pnt).any():
                user_points.append((pnt.y, pnt.x))  # Simpan (Lat, Lon)

        print(f"✅ Berhasil generate {len(user_points)} titik user di area pemukiman.")
        return user_points
    except Exception as e:
        print(f"❌ Error memuat pemukiman: {e}")
        return []


# --- 2. Fungsi Ambil Elevasi (Ketinggian) ---
def get_elevation(tif_path, lat, lon):
    try:
        with rasterio.open(tif_path) as src:
            # Cek CRS, jika beda lakukan transformasi koordinat
            if src.crs.to_string() != "EPSG:4326":
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat

            # Ambil nilai pixel
            vals = src.sample([(x, y)])
            elev = next(vals)[0]

            # Filter nilai nodata (biasanya minus besar)
            if elev < -100:
                return 0.0
            return float(elev)
    except:
        return 0.0
