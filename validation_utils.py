# validation_utils.py
import torch
import numpy as np
from torch.distributions import Categorical
from osrm_utils import get_osrm_distance_cached
from data_utils import get_adaptive_evac_candidates, is_coord_in_flood_zone


@torch.no_grad()
def evaluate_policy(model, user_coords, evac_candidates, flood_gdf, max_radius_km=5.0):
    """
    Mengevaluasi policy RL pada dataset validasi.
    Tidak ada gradient update (hanya forward pass).
    """
    model.eval()  # pastikan dalam mode evaluasi
    total_reward = 0.0
    total_distance = 0.0
    n_episodes = 0
    n_safe = 0

    print("\n🔍 Memulai evaluasi policy...\n")

    for user_lat, user_lon in user_coords:
        safe_candidates, _ = get_adaptive_evac_candidates(
            user_lat, user_lon, evac_candidates, flood_gdf, max_radius_km=max_radius_km
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
                    0.0,  # status bahaya
                ],
                dtype=torch.float32,
            ).unsqueeze(
                0
            )  # (1, 6)
            score = model(state).squeeze()
            scores.append(score)

        if not scores:
            continue

        probs = torch.softmax(torch.stack(scores), dim=0)
        m = Categorical(probs)
        action = m.sample()
        chosen_lat, chosen_lon = safe_candidates[action.item()]

        # Hitung metrik
        dist_km = get_osrm_distance_cached(user_lat, user_lon, chosen_lat, chosen_lon)
        is_flooded = is_coord_in_flood_zone(flood_gdf, chosen_lat, chosen_lon)

        # Reward validasi sederhana (sama rumusnya dgn training)
        if is_flooded:
            reward = -100.0
        elif dist_km <= 1.0:
            reward = 100.0
        elif dist_km <= 2.0:
            reward = 50.0
        elif dist_km <= 3.0:
            reward = 10.0
        else:
            reward = -20.0

        total_reward += reward
        total_distance += dist_km
        n_episodes += 1
        if not is_flooded:
            n_safe += 1

        print(
            f"User#{n_episodes:03d} | Aksi: ({chosen_lat:.5f},{chosen_lon:.5f}) | "
            f"Reward: {reward:.2f} | Jarak: {dist_km:.2f} km | Aman: {not is_flooded}"
        )

    if n_episodes == 0:
        print("⚠️ Tidak ada episode valid untuk evaluasi.")
        return None

    avg_reward = total_reward / n_episodes
    avg_distance = total_distance / n_episodes
    safe_ratio = n_safe / n_episodes * 100.0

    print("\n📊 Hasil Evaluasi:")
    print(f"  • Episode valid: {n_episodes}")
    print(f"  • Rata-rata Reward : {avg_reward:.2f}")
    print(f"  • Rata-rata Jarak  : {avg_distance:.2f} km")
    print(f"  • Persentase Aman  : {safe_ratio:.1f}%")

    return {
        "avg_reward": avg_reward,
        "avg_distance": avg_distance,
        "safe_ratio": safe_ratio,
        "n_episodes": n_episodes,
    }
