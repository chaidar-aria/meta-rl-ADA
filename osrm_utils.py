# osrm_utils.py
# Mengelola pengambilan data jarak dari OSRM dan sistem caching.

import json
import requests
from config import BASE_URL_OSRM

# Global cache untuk menyimpan hasil jarak OSRM
osrm_cache = {}


def load_osrm_cache(path="./osrm_cache.json"):
    """Memuat cache OSRM dari file JSON."""
    global osrm_cache
    try:
        with open(path, "r") as f:
            osrm_cache = json.load(f)
        print(f"✅ OSRM cache dimuat dari: {path} ({len(osrm_cache)} entri)")
    except FileNotFoundError:
        print(
            f"⚠️ File cache OSRM tidak ditemukan di {path}, memulai dari cache kosong."
        )
        osrm_cache = {}


def save_osrm_cache(path="./osrm_cache.json"):
    """Menyimpan cache OSRM ke file JSON."""
    with open(path, "w") as f:
        json.dump(osrm_cache, f, indent=2)
    print(f"💾 OSRM cache disimpan ke: {path}")


def get_osrm_distance_cached(start_lat, start_lon, end_lat, end_lon):
    """
    Mendapatkan jarak dari OSRM. Menggunakan cache jika tersedia,
    jika tidak, melakukan request baru dan menyimpan hasilnya ke cache.
    """
    global osrm_cache

    # Kunci cache dengan format standar (6 angka desimal)
    key = f"{start_lat:.6f},{start_lon:.6f}__{end_lat:.6f},{end_lon:.6f}"

    if key in osrm_cache:
        return osrm_cache[key]

    # Jika tidak ada di cache, lakukan request ke OSRM API
    try:
        url = f"{BASE_URL_OSRM}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=false"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        distance_km = data["routes"][0]["distance"] / 1000  # meter -> kilometer

        # Simpan hasil ke cache
        osrm_cache[key] = distance_km
        return distance_km
    except requests.RequestException as e:
        print(
            f"⚠️ Gagal menghubungi OSRM: {start_lat},{start_lon} -> {end_lat},{end_lon} | Error: {e}"
        )
        return float("inf")
    except (KeyError, IndexError):
        # Handle jika response OSRM tidak valid (misal: tidak ada rute)
        return float("inf")
