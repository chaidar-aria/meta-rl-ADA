# rl_logic_complete.py
from data_utils import is_coord_in_flood_zone
from osrm_utils import get_osrm_distance_cached


def compute_smart_reward(user_loc, evac_loc, flood_gdf, tif_path):
    """
    Menghitung reward berdasarkan:
    1. Banjir (Shapefile) -> Fatal Penalty
    2. Jarak (OSRM) -> Efficiency Reward
    3. Elevasi (TIF) -> Safety Bonus
    """
    user_lat, user_lon = user_loc
    evac_lat, evac_lon = evac_loc["coord"]  # Format dict baru dari GeoJSON

    # --- 1. CEK BANJIR (Constraint Mutlak) ---
    # Jika titik evakuasi ada di area genangan = HUKUMAN MATI (Reward sangat minus)
    if is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon):
        return -100.0, "Flooded 🌊"

    # --- 2. HITUNG JARAK OSRM (Efficiency) ---
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    # --- 3. CEK ELEVASI (Safety Bonus) ---
    # Kita import fungsi elevasi di dalam sini atau pass sebagai argumen
    from data_loader_complete import get_elevation

    u_elev = get_elevation(tif_path, user_lat, user_lon)
    e_elev = get_elevation(tif_path, evac_lat, evac_lon)
    elev_diff = e_elev - u_elev  # Positif jika evakuasi lebih tinggi

    # --- RUMUS REWARD ---
    reward = 0.0

    # Skor Jarak (Semakin dekat semakin besar nilainya)
    if dist_km <= 1.0:
        reward += 50
    elif dist_km <= 3.0:
        reward += 30
    elif dist_km <= 5.0:
        reward += 10
    else:
        reward -= dist_km * 2  # Penalti jika terlalu jauh

    # Skor Elevasi (Bonus jika lari ke tempat tinggi)
    if elev_diff > 0:
        reward += elev_diff * 2  # Dapat poin plus sesuai beda tinggi
    elif elev_diff < -1:
        reward -= 5  # Penalti kecil jika lari ke tempat lebih rendah (kecuali terpaksa)

    return reward, f"Dist:{dist_km:.1f}km | ElevDiff:{elev_diff:.1f}m"
