# rl_core.py
# Logika inti RL yang disesuaikan untuk skenario evakuasi banjir.

import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.distributions import Categorical

from device_config import DEVICE
from config import INNER_LR, INNER_STEPS, GAMMA
from data_utils import get_adaptive_evac_candidates, is_coord_in_flood_zone
from osrm_utils import get_osrm_distance_cached


def compute_flood_reward(user_lat, user_lon, evac_lat, evac_lon, flood_gdf):
    """
    Menghitung reward berdasarkan status banjir dan jarak.
    """
    is_flooded = is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon)
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    # Penalti sangat besar jika titik evakuasi ternyata banjir
    if is_flooded:
        return -100.0, 1, dist_km  # 1 menandakan "banjir"

    # Reward berdasarkan kedekatan jarak jika aman
    if dist_km <= 1.0:
        reward = 100.0
    elif dist_km <= 2.0:
        reward = 50.0
    elif dist_km <= 3.0:
        reward = 10.0
    else:
        reward = -20.0  # Penalti ringan untuk jarak yang terlalu jauh

    return reward, 0, dist_km  # 0 menandakan "aman"


def rollout(policy, user_coords, evac_candidates, flood_gdf):
    """
    Menjalankan episode untuk semua user dalam skenario banjir.
    """
    trajectories = []
    for user_lat, user_lon in user_coords:
        filtered_candidates, _ = get_adaptive_evac_candidates(
            user_lat, user_lon, evac_candidates, flood_gdf
        )
        if not filtered_candidates:
            continue

        scores = []
        for evac_lat, evac_lon in filtered_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

            # --- REPRESENTASI STATE BARU ---
            # Fitur ke-6 sekarang adalah 0 (aman), karena kita sudah memfilter
            # kandidat yang tidak aman sebelumnya.
            state = torch.tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,  # Normalisasi jarak
                    0.0,  # Status bahaya (0 = aman)
                ],
                dtype=torch.float32,
            ).to(DEVICE)
            scores.append(policy(state).squeeze())

        if not scores:
            continue

        scores_tensor = torch.stack(scores)
        probs = torch.softmax(scores_tensor, dim=0)

        if torch.any(torch.isnan(probs)) or torch.any(probs <= 0):
            continue

        m = Categorical(probs)
        action = m.sample()

        chosen_evac_lat, chosen_evac_lon = filtered_candidates[action.item()]

        # Panggil fungsi reward baru untuk skenario banjir
        reward, _, _ = compute_flood_reward(
            user_lat, user_lon, chosen_evac_lat, chosen_evac_lon, flood_gdf
        )

        trajectories.append({"log_prob": m.log_prob(action), "reward": reward})

    return trajectories


def compute_loss(log_probs, rewards):
    """
    Menghitung loss REINFORCE dari log-probabilities dan rewards.
    """
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    # Normalisasi returns untuk stabilitas training
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).sum()
    return loss


def adapt(model, user_coords, evac_candidates, flood_gdf):
    """
    Menjalankan beberapa langkah adaptasi (inner loop) pada salinan model
    untuk satu task spesifik (skenario banjir).
    """
    model_adapted = deepcopy(model)
    optimizer = optim.SGD(model_adapted.parameters(), lr=INNER_LR)

    for _ in range(INNER_STEPS):
        # Hasilkan trajektori menggunakan data banjir
        trajectories = rollout(model_adapted, user_coords, evac_candidates, flood_gdf)

        if not trajectories:
            continue  # Lewati step jika tidak ada data valid

        log_probs = [t["log_prob"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        loss = compute_loss(log_probs, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model_adapted
