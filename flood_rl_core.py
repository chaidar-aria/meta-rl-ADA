# flood_rl_core.py
# Berisi fungsi-fungsi inti untuk training RL standar pada skenario banjir.

import torch
from torch.distributions import Categorical

from device_config import DEVICE
from config import GAMMA
from data_utils import get_adaptive_evac_candidates
from osrm_utils import get_osrm_distance_cached
from data_utils import is_coord_in_flood_zone


def compute_flood_reward(user_lat, user_lon, evac_lat, evac_lon, flood_gdf):
    """Menghitung reward berdasarkan status banjir dan jarak."""
    is_flooded = is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon)
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    if is_flooded:
        return -100.0  # Penalti sangat besar jika titik evakuasi banjir

    # Reward berdasarkan kedekatan jarak jika aman
    if dist_km <= 1.0:
        return 100.0
    if dist_km <= 2.0:
        return 50.0
    if dist_km <= 3.0:
        return 10.0
    return -20.0  # Penalti ringan untuk jarak yang terlalu jauh


def rollout(policy, user_coords, evac_candidates, flood_gdf):
    """
    Menjalankan satu episode (satu 'rollout') untuk semua user dan mengumpulkan
    log probabilitas, serta reward.
    """
    trajectories = []
    for user_lat, user_lon in user_coords:
        # Dapatkan kandidat evakuasi yang aman saja
        safe_candidates, _ = get_adaptive_evac_candidates(
            user_lat, user_lon, evac_candidates, flood_gdf
        )
        if not safe_candidates:
            continue

        scores = []
        for evac_lat, evac_lon in safe_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            state = torch.tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,
                    0.0,  # Status bahaya (selalu 0 karena sudah difilter aman)
                ],
                dtype=torch.float32,
            ).to(DEVICE)
            scores.append(policy(state).squeeze())

        if not scores:
            continue

        probs = torch.softmax(torch.stack(scores), dim=0)
        if torch.any(torch.isnan(probs)) or torch.any(probs <= 0):
            continue

        m = Categorical(probs)
        action = m.sample()
        chosen_lat, chosen_lon = safe_candidates[action.item()]

        reward = compute_flood_reward(
            user_lat, user_lon, chosen_lat, chosen_lon, flood_gdf
        )
        trajectories.append({"log_prob": m.log_prob(action), "reward": reward})

    return trajectories


def compute_loss(log_probs, rewards):
    """Menghitung loss REINFORCE."""
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    return -(log_probs * returns).sum()
