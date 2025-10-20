# evaluation.py
# Berisi fungsi untuk mengevaluasi performa model.

import torch
import numpy as np
from data_utils import prepare_mmi_tensors, get_nearest_mmi_tensor
from osrm_utils import get_osrm_distance_cached
from rl_core import compute_reward
from device_config import DEVICE
from logging_utils import get_or_create_log_file, save_multi_event_log


def evaluate_model(model, user_coords, evac_candidates, mmi_coords, mmi_values):
    """
    Mengevaluasi model dengan memilih aksi terbaik (skor tertinggi) secara deterministik.
    """
    correct = 0
    results = []

    with torch.no_grad():
        for user_lat, user_lon in user_coords:
            best_score = -float("inf")
            best_action_details = None

            for evac_lat, evac_lon in evac_candidates:
                dist_km = get_osrm_distance_cached(
                    user_lat, user_lon, evac_lat, evac_lon
                )
                if not np.isfinite(dist_km):
                    continue

                mmi, _ = get_nearest_mmi_tensor(
                    evac_lat, evac_lon, mmi_coords, mmi_values
                )
                if not np.isfinite(mmi.item()):
                    continue

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

                score = model(state).item()

                if score > best_score:
                    best_score = score
                    best_action_details = (evac_lat, evac_lon)

            if best_action_details is None:
                continue

            best_evac_lat, best_evac_lon = best_action_details
            reward, mmi_val, dist_km = compute_reward(
                user_lat, user_lon, best_evac_lat, best_evac_lon, mmi_coords, mmi_values
            )

            is_valid = reward > 0  # Rute yang valid adalah yang reward-nya positif
            correct += int(is_valid)

            results.append(
                {
                    "user_coord": [user_lat, user_lon],
                    "evac_coord": [best_evac_lat, best_evac_lon],
                    "mmi": mmi_val,
                    "distance_km": dist_km,
                    "reward": reward,
                    "valid": int(is_valid),
                }
            )

    accuracy = (correct / len(user_coords)) if user_coords else 0.0
    return results, accuracy


def evaluate_only_and_log(
    model, user_coords, evac_candidates, mmi_points, event_id, output_dir="logs"
):
    """
    Hanya menjalankan evaluasi pada model yang diberikan (tanpa training) dan mencatat hasilnya.
    """
    mmi_coords, mmi_values = prepare_mmi_tensors(mmi_points)

    eval_results, accuracy = evaluate_model(
        model, user_coords, evac_candidates, mmi_coords, mmi_values
    )

    log_data = {
        "event_id": event_id,
        "success_rate": accuracy,
        "total_users": len(user_coords),
        "successful_routes": int(accuracy * len(user_coords)),
        "results": eval_results,
    }

    log_path = get_or_create_log_file(output_dir)
    save_multi_event_log(event_id, log_data, log_path)

    print(
        f"✅ Event '{event_id}' dievaluasi | Tingkat Keberhasilan: {accuracy:.3f} | Log di: '{log_path.name}'"
    )
    return accuracy
