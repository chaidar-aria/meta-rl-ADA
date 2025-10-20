# training.py
# Berisi fungsi utama untuk menjalankan proses meta-training.

import torch
import torch.optim as optim
from model import PolicyNetwork
from rl_core import adapt, rollout, compute_loss
from device_config import DEVICE
from config import LEARNING_RATE, MAX_META_ITER
from data_utils import prepare_mmi_tensors


def meta_train(tasks, save_path="trained_meta_model.pt", print_every=10):
    """
    Menjalankan proses meta-training MAML dengan REINFORCE.

    Args:
        tasks (list): List of task dictionaries.
        save_path (str): Path untuk menyimpan model terlatih.
        print_every (int): Frekuensi untuk mencetak log training.
    """
    meta_model = PolicyNetwork(input_dim=6, output_dim=1).to(DEVICE)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)

    for iteration in range(MAX_META_ITER):
        meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
        valid_task_count = 0
        total_loss = 0
        total_reward = 0

        for task in tasks:
            mmi_coords, mmi_values = prepare_mmi_tensors(task["mmi_points"])

            if mmi_coords.numel() == 0:
                continue

            model_adapted = adapt(
                meta_model,
                task["user_coords"],
                task["evac_candidates"],
                mmi_coords,
                mmi_values,
            )

            trajectories = rollout(
                model_adapted,
                task["user_coords"],
                task["evac_candidates"],
                mmi_coords,
                mmi_values,
            )

            if not trajectories:
                continue

            log_probs = [t["log_prob"] for t in trajectories]
            rewards = [t["reward"] for t in trajectories]
            loss = compute_loss(log_probs, rewards)

            grads = torch.autograd.grad(loss, model_adapted.parameters())
            for i, g in enumerate(grads):
                meta_grads[i] += g.detach()

            valid_task_count += 1
            total_loss += loss.item()
            total_reward += sum(rewards)

        if valid_task_count == 0:
            if (iteration % print_every) == 0:
                print(
                    f"[META {iteration}/{MAX_META_ITER}] Tidak ada trajektori valid pada iterasi ini."
                )
            continue

        # Update meta-model
        scale = 1.0 / float(valid_task_count)
        for p, g in zip(meta_model.parameters(), meta_grads):
            p.grad = g * scale
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if (iteration % print_every) == 0:
            avg_loss = total_loss / valid_task_count
            avg_reward = total_reward / valid_task_count
            print(
                f"[META {iteration}/{MAX_META_ITER}] Avg Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.2f}"
            )

    print("✅ Meta-training selesai.")
    torch.save(meta_model.state_dict(), save_path)
    print(f"💾 Model Meta-RL disimpan sebagai '{save_path}'")
    return meta_model
