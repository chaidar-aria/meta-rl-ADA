# training.py
# Berisi fungsi meta_train yang disesuaikan untuk skenario banjir.

import torch
import torch.optim as optim
import numpy as np

from device_config import DEVICE
from config import LEARNING_RATE, MAX_META_ITER
from model import PolicyNetwork
from rl_core import adapt, rollout, compute_loss


def meta_train(
    tasks, *, print_every: int = 10, save_path: str = "meta_model_banjir.pt"
):
    """
    Menjalankan proses meta-training untuk skenario banjir.

    - Input: tasks = list of dict, tiap dict berisi:
      {
        "event_id": str,
        "user_coords": List[Tuple[lat, lon]],
        "evac_candidates": List[Tuple[lat, lon]],
        "flood_polygons": GeoDataFrame,
      }
    """
    meta_model = PolicyNetwork(input_dim=6, output_dim=1).to(DEVICE)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)

    print(f"Memulai meta-training untuk {MAX_META_ITER} iterasi...")

    for iteration in range(MAX_META_ITER):
        meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
        valid_task_count = 0
        total_reward_iter = 0

        for task in tasks:
            # --- PERUBAHAN UTAMA: Gunakan data banjir ---
            flood_gdf = task["flood_polygons"]

            # 1. Adaptasi: Lakukan fine-tuning pada salinan model untuk task ini
            model_adapted = adapt(
                meta_model,
                task["user_coords"],
                task["evac_candidates"],
                flood_gdf,  # Menggunakan GeoDataFrame poligon banjir
            )

            # 2. Evaluasi: Jalankan rollout pada model yang sudah diadaptasi
            trajectories = rollout(
                model_adapted,
                task["user_coords"],
                task["evac_candidates"],
                flood_gdf,  # Menggunakan GeoDataFrame poligon banjir
            )

            if not trajectories:
                continue

            log_probs = [t["log_prob"] for t in trajectories]
            rewards = [t["reward"] for t in trajectories]
            loss = compute_loss(log_probs, rewards)

            # Hitung gradien dan akumulasikan
            grads = torch.autograd.grad(loss, model_adapted.parameters())
            for i, g in enumerate(grads):
                meta_grads[i] += g.detach()

            valid_task_count += 1
            total_reward_iter += np.sum(rewards)

        if valid_task_count == 0:
            if (iteration + 1) % print_every == 0:
                print(
                    f"[Iterasi {iteration+1}/{MAX_META_ITER}] Tidak ada trajektori valid yang dihasilkan."
                )
            continue

        # Terapkan gradien yang sudah dirata-ratakan ke meta-model
        scale = 1.0 / float(valid_task_count)
        for p, g in zip(meta_model.parameters(), meta_grads):
            p.grad = g * scale

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if (iteration + 1) % print_every == 0:
            avg_reward = (
                total_reward_iter / valid_task_count if valid_task_count > 0 else 0
            )
            print(
                f"[Iterasi {iteration+1}/{MAX_META_ITER}] Rata-rata Reward: {avg_reward:.2f}"
            )

    print("✅ Meta-training selesai.")
    torch.save(meta_model.state_dict(), save_path)
    print(f"💾 Model Meta-RL disimpan sebagai '{save_path}'")
    return meta_model
