# flood_trainer.py
# Berisi loop training RL standar dan fungsi evaluasi untuk skenario banjir.

import torch
import numpy as np
import json
from pathlib import Path

from device_config import DEVICE
from flood_rl_core import rollout, compute_loss
from osrm_utils import get_osrm_distance_cached
from data_utils import is_coord_in_flood_zone
from logging_utils import get_or_create_log_file


def train_rl_model(
    model, optimizer, num_episodes, user_coords, evac_candidates, flood_gdf
):
    """
    Fungsi utama untuk melatih model RL pada satu skenario banjir.
    """
    print(f"\n🚀 Memulai training untuk {num_episodes} episode...")
    model.train()  # Set model ke mode training

    for episode in range(num_episodes):
        # Jalankan rollout untuk mendapatkan pengalaman
        trajectories = rollout(model, user_coords, evac_candidates, flood_gdf)

        if not trajectories:
            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode+1}/{num_episodes} | Tidak ada trajektori valid."
                )
            continue

        log_probs = [t["log_prob"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        # Hitung loss dan update model
        loss = compute_loss(log_probs, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards)
            print(
                f"Episode {episode+1}/{num_episodes} | Rata-rata Reward: {avg_reward:.2f} | Loss: {loss.item():.2f}"
            )

    print("✅ Training selesai.")
    return model


def evaluate_model(
    model, user_coords, evac_candidates, flood_gdf, output_dir="hasil_evaluasi_banjir"
):
    """Mengevaluasi model yang sudah dilatih."""
    print("\n🔍 Mengevaluasi model terlatih...")
    model.eval()  # Set model ke mode evaluasi
    correct_predictions = 0

    with torch.no_grad():
        for user_lat, user_lon in user_coords:
            best_score = -float("inf")
            chosen_evac = None

            for evac_lat, evac_lon in evac_candidates:
                dist_km = get_osrm_distance_cached(
                    user_lat, user_lon, evac_lat, evac_lon
                )
                is_flooded = is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon)

                state = torch.tensor(
                    [
                        user_lat / 100.0,
                        user_lon / 100.0,
                        evac_lat / 100.0,
                        evac_lon / 100.0,
                        dist_km / 10.0,
                        1.0 if is_flooded else 0.0,
                    ],
                    dtype=torch.float32,
                ).to(DEVICE)

                score = model(state).item()

                if score > best_score:
                    best_score = score
                    chosen_evac = (evac_lat, evac_lon)

            if chosen_evac:
                is_safe = not is_coord_in_flood_zone(
                    flood_gdf, chosen_evac[0], chosen_evac[1]
                )
                if is_safe:
                    correct_predictions += 1

    accuracy = (correct_predictions / len(user_coords)) * 100 if user_coords else 0
    print(
        f"✅ Evaluasi Selesai | Akurasi Pilihan Aman: {accuracy:.2f}% ({correct_predictions}/{len(user_coords)})"
    )
    return accuracy
