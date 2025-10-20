# evaluation.py
# Menyediakan fungsi untuk evaluasi model pada skenario banjir.

import torch
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from device_config import DEVICE
from osrm_utils import get_osrm_distance_cached
from data_utils import is_coord_in_flood_zone
from rl_core import compute_flood_reward
from logging_utils import get_or_create_log_file


def evaluate_only_and_log(
    model,
    event_id,
    user_coords,
    evac_candidates,
    flood_polygons_gdf,
    output_dir="evaluation_logs_banjir",
):
    """
    Mengevaluasi model pada satu skenario banjir tanpa melakukan training,
    lalu menyimpan hasilnya ke dalam file log.
    """
    model.eval()  # Set model ke mode evaluasi
    correct_predictions = 0
    results_log = []

    print(f"--- Mengevaluasi Skenario: {event_id} ---")

    with torch.no_grad():
        for user_lat, user_lon in user_coords:
            best_score = -float("inf")
            best_action_details = None

            for evac_lat, evac_lon in evac_candidates:
                dist_km = get_osrm_distance_cached(
                    user_lat, user_lon, evac_lat, evac_lon
                )
                is_flooded = is_coord_in_flood_zone(
                    flood_polygons_gdf, evac_lat, evac_lon
                )

                # State yang diberikan ke model
                state = torch.tensor(
                    [
                        user_lat / 100.0,
                        user_lon / 100.0,
                        evac_lat / 100.0,
                        evac_lon / 100.0,
                        dist_km / 10.0,
                        1.0 if is_flooded else 0.0,  # 1 jika banjir, 0 jika aman
                    ],
                    dtype=torch.float32,
                ).to(DEVICE)

                score = model(state).item()

                if score > best_score:
                    best_score = score
                    best_action_details = {
                        "evac_lat": evac_lat,
                        "evac_lon": evac_lon,
                        "dist_km": dist_km,
                        "is_flooded": is_flooded,
                    }

            if best_action_details is None:
                continue

            # Cek apakah pilihan terbaik model adalah pilihan yang valid (aman)
            is_valid_choice = not best_action_details["is_flooded"]
            if is_valid_choice:
                correct_predictions += 1

            results_log.append(
                {
                    "user_coord": [user_lat, user_lon],
                    "chosen_evac": [
                        best_action_details["evac_lat"],
                        best_action_details["evac_lon"],
                    ],
                    "distance_km": best_action_details["dist_km"],
                    "is_flooded": best_action_details["is_flooded"],
                    "is_valid": is_valid_choice,
                }
            )

    # Hitung akurasi
    accuracy = (correct_predictions / len(user_coords)) * 100 if user_coords else 0
    print(
        f"Akurasi Pilihan Aman: {accuracy:.2f}% ({correct_predictions}/{len(user_coords)})"
    )

    # Simpan log
    log_file_path = get_or_create_log_file(output_dir)
    try:
        with open(log_file_path, "r") as f:
            all_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_logs = {}

    all_logs[event_id] = {
        "accuracy_percent": accuracy,
        "total_users": len(user_coords),
        "safe_choices": correct_predictions,
        "details": results_log,
    }

    with open(log_file_path, "w") as f:
        json.dump(all_logs, f, indent=4)
    print(f"📁 Log evaluasi untuk '{event_id}' disimpan di '{log_file_path}'")
