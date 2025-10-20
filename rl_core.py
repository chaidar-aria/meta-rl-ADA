# rl_core.py
# Berisi fungsi inti untuk proses Reinforcement Learning:
# - compute_reward: Menghitung reward berdasarkan aksi.
# - rollout: Menjalankan satu episode dan mengumpulkan trajektori.
# - compute_loss: Menghitung loss REINFORCE.
# - adapt: Melakukan adaptasi model pada satu task.

import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.distributions import Categorical

from device_config import DEVICE
from config import INNER_LR, INNER_STEPS, GAMMA
from data_utils import get_nearest_mmi_tensor, get_adaptive_evac_candidates
from osrm_utils import get_osrm_distance_cached


def compute_reward(user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values):
    """Menghitung reward berdasarkan MMI dan jarak OSRM."""
    mmi, _ = get_nearest_mmi_tensor(evac_lat, evac_lon, mmi_coords, mmi_values)
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    if not np.isfinite(dist_km) or not np.isfinite(mmi.item()):
        return -100.0, mmi.item(), dist_km

    if mmi >= 4.0:
        return -100.0, mmi.item(), dist_km

    if dist_km <= 1.0:
        reward = 100.0
    elif dist_km <= 2.0:
        reward = 50.0
    elif dist_km <= 3.0:
        reward = 10.0
    elif dist_km <= 4.0:
        reward = -10.0
    elif dist_km <= 5.0:
        reward = -25.0
    else:
        reward = -30.0

    return reward, mmi.item(), dist_km


def rollout(policy, user_coords, evac_candidates, mmi_coords, mmi_values):
    """Menjalankan episode untuk semua user dalam satu task dan mengumpulkan trajektori."""
    trajectories = []
    for user_lat, user_lon in user_coords:
        filtered_candidates, _ = get_adaptive_evac_candidates(
            user_lat,
            user_lon,
            evac_candidates,
            mmi_coords,
            mmi_values,
            max_radius_km=5.0,
        )
        if not filtered_candidates:
            continue

        scores = []
        for evac_lat, evac_lon in filtered_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            mmi, _ = get_nearest_mmi_tensor(evac_lat, evac_lon, mmi_coords, mmi_values)
            state = torch.tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,
                    mmi.item() / 10.0,
                ],
                dtype=torch.float32,
            ).to(DEVICE)
            scores.append(policy(state).squeeze())

        if not scores:
            continue

        scores = torch.stack(scores)
        probs = torch.softmax(scores, dim=0)

        if torch.any(torch.isnan(probs)) or torch.any(probs <= 0):
            continue

        m = Categorical(probs)
        action = m.sample()

        chosen_evac_lat, chosen_evac_lon = filtered_candidates[action.item()]
        reward, _, _ = compute_reward(
            user_lat, user_lon, chosen_evac_lat, chosen_evac_lon, mmi_coords, mmi_values
        )

        trajectories.append({"log_prob": m.log_prob(action), "reward": reward})

    return trajectories


def compute_loss(log_probs, rewards):
    """Menghitung loss REINFORCE dari trajektori."""
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).sum()
    return loss


def adapt(model, user_coords, evac_candidates, mmi_coords, mmi_values):
    """Melakukan inner-loop update (adaptasi) pada model untuk task tertentu."""
    model_adapted = deepcopy(model)
    optimizer = optim.SGD(model_adapted.parameters(), lr=INNER_LR)

    for _ in range(INNER_STEPS):
        trajectories = rollout(
            model_adapted, user_coords, evac_candidates, mmi_coords, mmi_values
        )
        if not trajectories:
            continue

        log_probs = [t["log_prob"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        loss = compute_loss(log_probs, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model_adapted
