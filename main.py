# main.py
# Titik masuk utama untuk menjalankan training dan evaluasi model Meta-RL.

import torch
import numpy as np

# Impor dari modul-modul yang sudah dibuat
from device_config import info_device, DEVICE
from model import PolicyNetwork
from osrm_utils import load_osrm_cache, save_osrm_cache
from training import meta_train
from evaluation import evaluate_only_and_log


def create_dummy_tasks(num_tasks=5, users_per_task=10, candidates_per_task=20):
    """Membuat data dummy untuk demonstrasi."""
    tasks = []
    for i in range(num_tasks):
        # Koordinat di sekitar area tertentu (misal: Jawa Barat)
        user_coords = (np.random.rand(users_per_task, 2) * np.array([1, 2])) + np.array(
            [-6.9, 107.6]
        )
        evac_candidates = (
            np.random.rand(candidates_per_task, 2) * np.array([0.1, 0.2])
        ) + np.array([-6.9, 107.6])
        mmi_points = [
            {"lat": lat, "lon": lon, "mmi": np.random.uniform(1, 5)}
            for lat, lon in (np.random.rand(50, 2) * np.array([0.2, 0.4]))
            + np.array([-7.0, 107.5])
        ]

        tasks.append(
            {
                "event_id": f"dummy_event_{i+1}",
                "user_coords": user_coords.tolist(),
                "evac_candidates": evac_candidates.tolist(),
                "mmi_points": mmi_points,
            }
        )
    return tasks


def main():
    """Fungsi utama untuk menjalankan alur kerja."""
    # 1. Tampilkan info perangkat
    info_device()

    # 2. Muat cache OSRM jika ada
    load_osrm_cache("osrm_cache.json")

    # 3. Buat data dummy untuk training dan evaluasi
    print("\nMembuat data dummy untuk tasks...")
    tasks = create_dummy_tasks(num_tasks=3, users_per_task=5)

    # 4. Jalankan Meta-Training
    print("\nMemulai proses meta-training...")
    trained_model = meta_train(tasks, save_path="meta_model_final.pt")

    # 5. Jalankan Evaluasi pada model yang sudah dilatih
    print("\nMemulai proses evaluasi...")
    # Anda bisa menggunakan task yang sama atau task baru untuk evaluasi
    evaluation_task = create_dummy_tasks(num_tasks=1)[0]

    evaluate_only_and_log(
        model=trained_model,
        user_coords=evaluation_task["user_coords"],
        evac_candidates=evaluation_task["evac_candidates"],
        mmi_points=evaluation_task["mmi_points"],
        event_id=evaluation_task["event_id"],
        output_dir="evaluation_logs",
    )

    # 6. Simpan cache OSRM yang mungkin sudah diperbarui
    save_osrm_cache("osrm_cache.json")
    print("\nProses selesai.")


if __name__ == "__main__":
    main()
