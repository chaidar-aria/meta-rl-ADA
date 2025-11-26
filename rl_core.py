import torch
from torch.distributions import Categorical
from device_config import DEVICE
from osrm_utils import get_osrm_distance_cached
from data_utils import is_coord_in_flood_zone, get_elevation_at_point


def compute_multi_objective_reward(
    user_lat, user_lon, evac_lat, evac_lon, flood_gdf, user_elev=0.0, evac_elev=0.0
):
    """
    REVISI LOGIKA: LEBIH STRICT TERHADAP JARAK.
    """
    # 1. CEK BANJIR (CONSTRAINT MUTLAK)
    if is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon):
        return -100.0, True, 0.0

    # 2. HITUNG JARAK OSRM
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    # 3. HITUNG BEDA ELEVASI
    elev_diff = evac_elev - user_elev

    reward = 0.0

    # --- A. SKOR JARAK (PENALTI DIPERBERAT) ---
    if dist_km <= 0.5:
        reward += 100.0  # Sangat dekat (Emas)
    elif dist_km <= 1.0:
        reward += 70.0  # Dekat (Perak)
    elif dist_km <= 1.5:
        reward += 30.0  # Lumayan
    elif dist_km <= 2.0:
        reward += 5.0  # Batas toleransi
    else:
        # Jarak > 2km langsung kena hukuman berat!
        # Rumus: -10 poin per km tambahannya
        reward -= dist_km * 15.0

    # --- B. SKOR ELEVASI (DIBATASI) ---
    # Kita batasi bonus elevasi agar tidak mengalahkan faktor jarak
    # Maksimal bonus elevasi di-cap (misal max 20 poin)
    if elev_diff > 0:
        bonus = elev_diff * 2.0  # Faktor pengali dikurangi jadi 2.0
        reward += min(bonus, 30.0)  # Capping: Maksimal bonus cuma 30
    elif elev_diff < -1.0:
        reward -= 10.0  # Penalti jika lari ke tempat lebih rendah

    return reward, False, dist_km


def rollout(policy, user_coords, evac_candidates, flood_gdf, tif_path):
    """
    Revisi Rollout: Membatasi radius pencarian agar tidak melihat yang jauh-jauh.
    """
    trajectories = []

    for user_lat, user_lon in user_coords:
        user_elev = get_elevation_at_point(tif_path, user_lat, user_lon)

        inputs = []
        valid_indices = []

        for idx, cand in enumerate(evac_candidates):
            e_lat, e_lon = cand["coord"]
            dist = get_osrm_distance_cached(user_lat, user_lon, e_lat, e_lon)

            # --- FILTER KERAS ---
            # Jangan biarkan agen melihat kandidat yang jaraknya > 3 km
            # Biar dia fokus mencari solusi lokal.
            if dist > 3.0:
                continue

            # Feature State: [u_lat, u_lon, e_lat, e_lon, dist, 0]
            inputs.append(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    e_lat / 100.0,
                    e_lon / 100.0,
                    dist / 10.0,
                    0.0,
                ]
            )
            valid_indices.append(idx)

        # Jika tidak ada kandidat dalam radius 3km, terpaksa cari yang agak jauh (fallback)
        if not inputs:
            for idx, cand in enumerate(evac_candidates):
                e_lat, e_lon = cand["coord"]
                dist = get_osrm_distance_cached(user_lat, user_lon, e_lat, e_lon)
                if dist > 10.0:
                    continue  # Limit absolut 10km
                inputs.append(
                    [
                        user_lat / 100.0,
                        user_lon / 100.0,
                        e_lat / 100.0,
                        e_lon / 100.0,
                        dist / 10.0,
                        0.0,
                    ]
                )
                valid_indices.append(idx)

        if not inputs:
            continue

        input_tensor = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)

        # ... (Sisa kode sama) ...
        scores = policy(input_tensor).squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        # Safety check untuk NaN scores
        if torch.isnan(scores).any():
            continue

        probs = torch.softmax(scores, dim=0)

        # Safety check untuk NaN probs
        if torch.isnan(probs).any():
            continue

        m = Categorical(probs)
        action_idx = m.sample()

        real_idx = valid_indices[action_idx.item()]
        chosen = evac_candidates[real_idx]
        e_lat, e_lon = chosen["coord"]
        e_elev = chosen["elev"]

        reward, is_flood, dist = compute_multi_objective_reward(
            user_lat,
            user_lon,
            e_lat,
            e_lon,
            flood_gdf,
            user_elev=user_elev,
            evac_elev=e_elev,
        )

        trajectories.append(
            {
                "log_prob": m.log_prob(action_idx),
                "reward": reward,
                "details": {
                    "name": chosen["name"],
                    "dist": dist,
                    "elev_diff": e_elev - user_elev,
                },
            }
        )

    return trajectories


# ... (biarkan fungsi compute_loss sama) ...
def compute_loss(log_probs, rewards):
    """Menghitung loss function (Standard REINFORCE)."""
    loss = 0
    returns = []
    R = 0

    # Calculate returns (cumulative reward discounted)
    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)

    returns = torch.tensor(returns).to(DEVICE)
    # Normalisasi returns (PENTING AGAR STABIL)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss_list = []
    for log_prob, R in zip(log_probs, returns):
        loss_list.append(-log_prob * R)

    if loss_list:
        loss = torch.stack(loss_list).sum()

    return loss
