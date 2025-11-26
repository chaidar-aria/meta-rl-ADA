# flood_trainer.py
import torch
import numpy as np
from rl_core import rollout, compute_loss


def train_rl_model(
    model, optimizer, num_episodes, user_coords, evac_candidates, flood_gdf, tif_path
):
    """
    Loop utama training RL.

    Args:
        evac_candidates: List of dict [{'coord':..., 'name':..., 'elev':...}]
                         (Harus sudah ada key 'elev' dari pre-processing di notebook)
        tif_path: Path ke file elevasi (untuk menghitung elevasi user yg random)
    """
    print(f"\n🚀 Memulai Training: {num_episodes} Episode")
    print(f"   ℹ️ Data: {len(user_coords)} Users, {len(evac_candidates)} Titik Evakuasi")

    model.train()
    history_rewards = []

    for episode in range(num_episodes):
        # 1. Jalankan Rollout (Simulasi 1 set user)
        trajectories = rollout(model, user_coords, evac_candidates, flood_gdf, tif_path)

        if not trajectories:
            print(f"   ⚠️ Episode {episode+1}: Tidak ada trajektori valid.")
            continue

        # 2. Ambil Log Probabilitas & Reward
        log_probs = [t["log_prob"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        # 3. Hitung Loss & Update Model
        loss = compute_loss(log_probs, rewards)

        optimizer.zero_grad()
        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
        else:
            loss_val = 0.0

        # 4. Logging & Statistik
        avg_reward = np.mean(rewards)
        history_rewards.append(avg_reward)

        # Log setiap 5 episode
        if (episode + 1) % 5 == 0:
            # Ambil contoh aksi terakhir untuk dicek manusia
            last_info = trajectories[-1]["details"]
            print(
                f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Loss: {loss_val:.2f}"
            )
            print(f"   ↳ Contoh: User lari ke '{last_info['name']}'")
            print(
                f"      (Jarak: {last_info['dist']:.2f}km, Beda Elevasi: {last_info['elev_diff']:.1f}m)"
            )

    print("\n✅ Training Selesai.")
    return model, history_rewards
