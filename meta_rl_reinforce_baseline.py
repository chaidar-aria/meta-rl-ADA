# meta_rl_reinforce_fixed.py
# -----------------------------------------
# RL dengan reward lebih tajam & evaluasi akurasi

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import json
from device_config import DEVICE, info_device
from copy import deepcopy
from config import (
    INNER_LR,
    INNER_STEPS,
    GAMMA,
    MAX_META_ITER,
    EPSILON,
    LEARNING_RATE,
    BASE_URL_OSRM,
)
from datetime import datetime
from pathlib import Path
import json
import requests

info_device()
osrm_cache = {}  # Untuk cache jarak OSRM
# Variabel global: diinisialisasi sekali di awal run
RUN_LOG_PATH = None


# ===== Policy Network =====
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=1):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.out(x)


def prepare_mmi_tensors(mmi_points):
    coords = torch.tensor(
        [[p["lat"], p["lon"]] for p in mmi_points], dtype=torch.float32
    ).to(DEVICE)
    values = torch.tensor([p["mmi"] for p in mmi_points], dtype=torch.float32).to(
        DEVICE
    )
    return coords, values


def get_nearest_mmi_tensor(lat, lon, mmi_coords, mmi_values):
    user = torch.tensor([lat, lon], dtype=torch.float32).to(DEVICE)
    dists = torch.norm(mmi_coords - user, dim=1)
    idx = torch.argmin(dists)
    return mmi_values[idx], dists[idx] * 111.0


def load_osrm_cache(path="./osrm_cache/osrm_cache_from_shp.json"):
    global osrm_cache
    try:
        with open(path, "r") as f:
            osrm_cache = json.load(f)
        print(f"✅ OSRM cache dimuat dari: {path} ({len(osrm_cache)} entri)")
    except FileNotFoundError:
        print("⚠️ File cache tidak ditemukan, mulai dari kosong.")
        osrm_cache = {}


def get_osrm_distance_cached(start_lat, start_lon, end_lat, end_lon):
    global osrm_cache

    # Format-formatted key variations
    candidates = [
        f"{start_lat},{start_lon}__{end_lat},{end_lon}",  # old style
        f"{start_lat:.6f},{start_lon:.6f}__{end_lat:.6f},{end_lon:.6f}",
        f"{round(start_lat, 5)},{round(start_lon, 5)}__{round(end_lat, 5)},{round(end_lon, 5)}",
        f"{round(start_lat, 4)},{round(start_lon, 4)}__{round(end_lat, 4)},{round(end_lon, 4)}",
        # 🔥 New: parentheses style
        f"({start_lat},{start_lon})-({end_lat},{end_lon})",
        f"({start_lat:.6f},{start_lon:.6f})-({end_lat:.6f},{end_lon:.6f})",
        f"({round(start_lat, 5)},{round(start_lon, 5)})-({round(end_lat, 5)},{round(end_lon, 5)})",
    ]

    for key in candidates:
        if key in osrm_cache:
            return osrm_cache[key]

    # 🧯 Jika tidak ketemu → fallback request ke OSRM live
    try:
        url = f"{BASE_URL_OSRM}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=false"
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        distance_km = data["routes"][0]["distance"] / 1000  # meter → kilometer

        # 📝 Simpan ke cache dengan format standar (6 desimal)
        key = f"{start_lat:.6f},{start_lon:.6f}__{end_lat:.6f},{end_lon:.6f}"
        osrm_cache[key] = distance_km

        return distance_km

    except Exception as e:
        print(
            f"⚠️ Fallback OSRM gagal: {start_lat},{start_lon} → {end_lat},{end_lon} | {e}"
        )
        return float("inf")


def compute_reward(user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values):
    mmi, _ = get_nearest_mmi_tensor(evac_lat, evac_lon, mmi_coords, mmi_values)
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    # print(f"[DEBUG] MMI: {mmi:.1f} | Distance: {dist_km:.2f}")  # Logging tetap

    if not np.isfinite(dist_km) or not np.isfinite(mmi):
        return -100.0, mmi, dist_km

    if mmi >= 4.0:
        return -100.0, mmi, dist_km

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
        reward = -30.0  # 💡 Dulu -50, sekarang agak lunak agar model masih bisa belajar

    # print(f"[DEBUG] => Final reward: {reward}")
    return reward, mmi, dist_km


def get_adaptive_evac_candidates(
    user_lat, user_lon, all_candidates, mmi_coords, mmi_values, max_radius_km=5.0
):
    # Buat langkah radius hingga max_radius_km (dalam km)
    radius_steps = [1.0, 2.0, 3.0, 4.0, max_radius_km]

    for radius in radius_steps:
        filtered = []
        for evac_lat, evac_lon in all_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            if not np.isfinite(dist_km) or dist_km > radius:
                continue

            mmi, _ = get_nearest_mmi_tensor(evac_lat, evac_lon, mmi_coords, mmi_values)
            if not np.isfinite(mmi) or mmi >= 4:
                continue

            filtered.append((evac_lat, evac_lon, dist_km, mmi))

        if filtered:
            # Urutkan berdasarkan kombinasi: jarak dekat & MMI kecil
            filtered.sort(key=lambda x: (x[2], x[3]))  # sort by dist_km, then MMI
            coords_only = [(lat, lon) for lat, lon, _, _ in filtered]
            return coords_only, radius

    return [], None


def rollout(policy, user_coords, evac_candidates, mmi_coords, mmi_values):
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
            print("⚠️ Tidak ada kandidat valid untuk user ini (jauh atau zona bahaya).")
            continue

        scores = []
        valid_indices = []

        for idx, (evac_lat, evac_lon) in enumerate(filtered_candidates):
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            mmi, _ = get_nearest_mmi_tensor(evac_lat, evac_lon, mmi_coords, mmi_values)

            state = torch.tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,
                    mmi / 10.0,
                ],
                dtype=torch.float32,
            ).to(DEVICE)

            score = policy(state).squeeze()
            scores.append(score)
            valid_indices.append(idx)

        if not scores:
            continue

        scores = torch.stack(scores)
        if scores.dim() > 1:
            scores = scores.view(-1)  # Pastikan 1D

        scores = torch.clamp(scores, min=-100.0, max=100.0)
        probs = torch.softmax(scores, dim=0)

        if torch.any(torch.isnan(probs)) or torch.any(probs <= 0):
            print("⚠️ Probabilitas tidak valid, skip user ini.")
            continue

        m = torch.distributions.Categorical(probs)
        action = m.sample()

        # 🔧 Pastikan scalar
        if action.dim() > 0:
            action = action.item()

        action_idx = valid_indices[action]
        evac_lat, evac_lon = filtered_candidates[action_idx]

        reward, mmi_val, dist_km = compute_reward(
            user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values
        )

        trajectories.append(
            {"log_prob": m.log_prob(torch.tensor(action)), "reward": reward}
        )
        print(
            f"Episode: {len(trajectories)} | Aksi: {action_idx} → Reward: {reward:.2f} | MMI: {mmi_val:.2f} | Jarak: {dist_km:.2f} km"
        )

    if trajectories:
        reward_vals = [t["reward"] for t in trajectories]
        print(
            f"🎯 Distribusi reward (min/avg/max): {min(reward_vals):.2f} / {np.mean(reward_vals):.2f} / {max(reward_vals):.2f}"
        )

    return trajectories


def compute_loss(log_probs, rewards):
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).sum()
    return loss


def adapt(model, user_coords, evac_candidates, mmi_coords, mmi_values):
    model_adapted = deepcopy(model)
    optimizer = optim.SGD(model_adapted.parameters(), lr=INNER_LR)

    for _ in range(INNER_STEPS):
        trajectories = rollout(
            model_adapted, user_coords, evac_candidates, mmi_coords, mmi_values
        )

        if not trajectories:
            continue  # skip kalau tidak ada langkah valid

        log_probs = [t["log_prob"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]

        loss = compute_loss(log_probs, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model_adapted


# def meta_train(tasks):
#     meta_model = PolicyNetwork(input_dim=6, output_dim=1).to(DEVICE)
#     meta_optimizer = optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)

#     for iteration in range(MAX_META_ITER):
#         meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
#         for task in tasks:
#             model_adapted = adapt(
#                 meta_model,
#                 task["user_coords"],
#                 task["evac_candidates"],
#                 task["mmi_coords"],
#                 task["mmi_values"],
#             )

#             trajectories = rollout(
#                 model_adapted,
#                 task["user_coords"],
#                 task["evac_candidates"],
#                 task["mmi_coords"],
#                 task["mmi_values"],
#             )

#             if not trajectories:
#                 continue  # skip kalau rollout tidak menghasilkan data

#             log_probs = [t["log_prob"] for t in trajectories]
#             rewards = [t["reward"] for t in trajectories]
#             loss = compute_loss(log_probs, rewards)

#             grads = torch.autograd.grad(loss, model_adapted.parameters())
#             for i, g in enumerate(grads):
#                 meta_grads[i] += g.detach()

#         for p, g in zip(meta_model.parameters(), meta_grads):
#             p.grad = g / len(tasks)
#         meta_optimizer.step()
#         meta_optimizer.zero_grad()

#         if iteration % 50 == 0:
#             print(
#                 f"[META {iteration}] Contoh reward: {rewards[:3]} | Loss: {loss.item():.4f}"
#             )

#     print("✅ Meta-training selesai")
#     torch.save(meta_model.state_dict(), "trained_meta_model.pt")
#     print("💾 Model Meta-RL disimpan sebagai 'trained_meta_model.pt'")

#     return meta_model

def meta_train(tasks, *, print_every: int = 50, save_path: str = "trained_meta_model.pt"):
    """
    Drop-in meta_train untuk pipeline kamu:
    - Input: tasks = list of dict, tiap dict minimal berisi:
        {
          "event_id": str,
          "user_coords": List[Tuple[lat, lon]],
          "evac_candidates": List[Tuple[lat, lon]],
          # Pilih salah satu:
          # (1) Tensor siap pakai:
          "mmi_coords": torch.FloatTensor[N, 2],
          "mmi_values": torch.FloatTensor[N],
          # (2) Atau masih list dict:
          "mmi_points": [{"lat": ..., "lon": ..., "mmi": ...}, ...]
        }
    - Meng-handle tugas yang rollout-nya kosong (skip) tanpa mematikan training loop.
    - Memindahkan tensor ke DEVICE dan memastikan dtype float32.
    """
    import torch
    import torch.optim as optim

    def _coerce_mmi(task):
        """Kembalikan (mmi_coords[N,2], mmi_values[N]) dalam float32 di DEVICE."""
        if ("mmi_coords" in task) and ("mmi_values" in task):
            mmi_coords = task["mmi_coords"]
            mmi_values = task["mmi_values"]
            # pastikan tipe/perangkat
            if not torch.is_tensor(mmi_coords):
                mmi_coords = torch.tensor(mmi_coords, dtype=torch.float32)
            if not torch.is_tensor(mmi_values):
                mmi_values = torch.tensor(mmi_values, dtype=torch.float32)
            mmi_coords = mmi_coords.to(dtype=torch.float32, device=DEVICE)
            mmi_values = mmi_values.to(dtype=torch.float32, device=DEVICE)
            return mmi_coords, mmi_values

        # fallback: build dari mmi_points
        pts = task.get("mmi_points", [])
        if len(pts) == 0:
            # kembalikan tensor kosong agar adapt/rollout bisa memutuskan skip
            return (torch.empty((0, 2), dtype=torch.float32, device=DEVICE),
                    torch.empty((0,), dtype=torch.float32, device=DEVICE))

        latlon = [[float(p["lat"]), float(p["lon"])] for p in pts]
        vals = [float(p["mmi"]) for p in pts]
        mmi_coords = torch.tensor(latlon, dtype=torch.float32, device=DEVICE)
        mmi_values = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
        return mmi_coords, mmi_values

    # ===== init meta model & optimizer =====
    meta_model = PolicyNetwork(input_dim=6, output_dim=1).to(DEVICE)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)

    last_print = {"rewards": None, "loss": None, "event_id": None}

    for iteration in range(MAX_META_ITER):
        # siapkan buffer grad dengan shape persis param meta_model
        meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
        valid_task_count = 0

        for task in tasks:
            # pastikan mmi tensor on-device
            mmi_coords, mmi_values = _coerce_mmi(task)

            # jalankan adapt pada salinan meta_model (sesuai implementasi kamu)
            model_adapted = adapt(
                meta_model,
                task["user_coords"],
                task["evac_candidates"],
                mmi_coords,
                mmi_values,
            )

            # rollout pada model yang sudah diadaptasi
            trajectories = rollout(
                model_adapted,
                task["user_coords"],
                task["evac_candidates"],
                mmi_coords,
                mmi_values,
            )
            if not trajectories:
                continue  # skip task tanpa data

            log_probs = [t["log_prob"] for t in trajectories]
            rewards = [t["reward"] for t in trajectories]
            loss = compute_loss(log_probs, rewards)

            grads = torch.autograd.grad(loss, model_adapted.parameters())
            for i, g in enumerate(grads):
                meta_grads[i] += g.detach()

            valid_task_count += 1
            last_print.update({"rewards": rewards, "loss": loss, "event_id": task.get("event_id")})

        # tidak ada task valid -> lanjut ke iterasi berikutnya tanpa step
        if valid_task_count == 0:
            if (iteration % print_every) == 0:
                print(f"[META {iteration}] (no valid trajectories this iter)")
            continue

        # rata-ratakan grad, apply ke meta_model
        scale = 1.0 / float(valid_task_count)
        for p, g in zip(meta_model.parameters(), meta_grads):
            p.grad = g * scale
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if (iteration % print_every) == 0:
            if last_print["loss"] is not None:
                r = last_print["rewards"][:3] if last_print["rewards"] else []
                ev = last_print["event_id"]
                print(f"[META {iteration}] evt={ev} sample_rewards={r} | loss={last_print['loss'].item():.4f}")
            else:
                print(f"[META {iteration}] (no valid trajectories yet)")

    print("✅ Meta-training selesai")
    torch.save(meta_model.state_dict(), save_path)
    print(f"💾 Model Meta-RL disimpan sebagai '{save_path}'")
    return meta_model



def evaluate_model(model, user_coords, evac_candidates, mmi_coords, mmi_values):
    correct = 0
    results = []

    with torch.no_grad():
        for user_lat, user_lon in user_coords:
            best_score = -float("inf")
            best_action = None
            best_result = None

            for idx, (evac_lat, evac_lon) in enumerate(evac_candidates):
                dist_km = get_osrm_distance_cached(
                    user_lat, user_lon, evac_lat, evac_lon
                )
                mmi, _ = get_nearest_mmi_tensor(
                    evac_lat, evac_lon, mmi_coords, mmi_values
                )

                # ✅ Normalisasi 6 fitur input (konsisten dengan training)
                state = torch.tensor(
                    [
                        user_lat / 100.0,
                        user_lon / 100.0,
                        evac_lat / 100.0,
                        evac_lon / 100.0,
                        dist_km / 10.0,
                        mmi / 10.0,
                    ],
                    dtype=torch.float32,
                ).to(DEVICE)

                score = model(state).item()
                if score > best_score:
                    best_score = score
                    best_action = idx
                    best_result = (evac_lat, evac_lon, mmi, dist_km)

            if best_result is None:
                continue

            evac_lat, evac_lon, mmi_val, dist_km = best_result
            reward, mmi_val, dist_km = compute_reward(
                user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values
            )

            valid_mmi = int(mmi_val < 4.0)
            valid_dist = int(dist_km <= 1.0)
            is_valid = int(valid_mmi and valid_dist)

            results.append(
                {
                    "user_coord": [user_lat, user_lon],
                    "evac_coord": [evac_lat, evac_lon],
                    "mmi": float(mmi_val),
                    "reward": float(reward),
                    "distance_km": float(dist_km),
                    "valid_mmi": valid_mmi,
                    "valid_distance": valid_dist,
                    "valid": is_valid,
                }
            )
            correct += is_valid

    accuracy = correct / len(user_coords)
    return results, accuracy


# Gabungkan ke log global multi-event


def save_to_multi_event_log(event_id, logs, base_dir="train_log"):
    # 🗓️ Ambil tanggal hari ini (misal: 2025-06-07)
    today_str = datetime.today().strftime("%Y-%m-%d")
    save_dir = Path(base_dir) / today_str
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / "rl_logs_multi_event.json"

    # 🔄 Load jika file sudah ada
    if output_path.exists():
        with open(output_path, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    # ✍️ Tambahkan/update event log
    all_logs[event_id] = logs

    # 💾 Simpan kembali
    with open(output_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"📁 Log event '{event_id}' disimpan ke '{output_path}'")


# ===== Training per Event =====
def adapt_and_log(
    meta_model,
    user_coords,
    evac_candidates,
    mmi_points,
    event_id,
    output_path="rl_logs_meta_event.json",
):
    from copy import deepcopy

    model = deepcopy(meta_model)
    optimizer = optim.SGD(model.parameters(), lr=INNER_LR)
    logs = {"episodes": [], "final_evaluation": [], "action_distribution": {}}
    mmi_coords, mmi_values = prepare_mmi_tensors(mmi_points)

    # Inisialisasi jumlah kandidat akhir yang mungkin dipilih
    action_counter = [0] * len(evac_candidates)

    for episode in range(INNER_STEPS):
        log_probs, rewards = [], []
        episode_log = {"steps": [], "total_reward": 0, "loss": 0}

        for user_lat, user_lon in user_coords:
            filtered_candidates, _ = get_adaptive_evac_candidates(
                user_lat, user_lon, evac_candidates, mmi_coords, mmi_values
            )
            if not filtered_candidates:
                print(
                    f"⚠️ User ({user_lat:.4f}, {user_lon:.4f}) tidak punya kandidat valid."
                )
                continue

            scores = []
            for evac_lat, evac_lon in filtered_candidates:
                dist_km = get_osrm_distance_cached(
                    user_lat, user_lon, evac_lat, evac_lon
                )
                mmi, _ = get_nearest_mmi_tensor(
                    evac_lat, evac_lon, mmi_coords, mmi_values
                )

                state = torch.tensor(
                    [
                        user_lat / 100.0,
                        user_lon / 100.0,
                        evac_lat / 100.0,
                        evac_lon / 100.0,
                        dist_km / 10.0,
                        mmi / 10.0,
                    ],
                    dtype=torch.float32,
                ).to(DEVICE)

                score = model(state)
                scores.append(score)

            scores = torch.stack(scores)
            probs = torch.softmax(scores.view(-1), dim=0)
            dist = Categorical(probs)

            if np.random.rand() < EPSILON:
                action = torch.tensor(np.random.choice(len(filtered_candidates)))
            else:
                action = dist.sample()

            action_idx = int(action.item())
            evac_lat, evac_lon = filtered_candidates[action_idx]
            reward, mmi_val, dist_km = compute_reward(
                user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values
            )

            # Cari index kandidat asli dari evac_candidates
            try:
                global_action_idx = evac_candidates.index((evac_lat, evac_lon))
                action_counter[global_action_idx] += 1
            except ValueError:
                pass  # bisa diabaikan jika titik tidak ditemukan secara eksak

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            episode_log["steps"].append(
                {
                    "user_coord": [user_lat, user_lon],
                    "evac_coord": [evac_lat, evac_lon],
                    "mmi": float(mmi_val),
                    "reward": float(reward),
                    "action": (
                        global_action_idx if "global_action_idx" in locals() else -1
                    ),
                    "distance_km": float(dist_km),
                }
            )

        if not log_probs:
            print(
                f"⚠️ Episode {episode+1} diabaikan: tidak ada langkah valid.", flush=True
            )
            continue

        returns, G = [], 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(log_probs) * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_log["total_reward"] = float(sum(rewards))
        episode_log["loss"] = float(loss)
        logs["episodes"].append(episode_log)

        print(
            f"🌀 Episode {episode+1}/{INNER_STEPS} | Reward: {episode_log['total_reward']:.2f} | Loss: {loss:.4f}",
            flush=True,
        )

    # === Evaluasi akhir
    eval_results, accuracy = evaluate_model(
        model, user_coords, evac_candidates, mmi_coords, mmi_values
    )
    logs["final_evaluation"] = eval_results
    logs["accuracy"] = accuracy
    logs["action_distribution"] = {
        str(i): count for i, count in enumerate(action_counter)
    }

    save_to_multi_event_log(event_id, logs, output_path)
    print(
        f"✅ Evaluasi selesai untuk event '{event_id}' | Akurasi: {accuracy:.3f}\n",
        flush=True,
    )

    return logs


def get_or_create_log_file(output_dir="log_eval_and_reward"):
    global RUN_LOG_PATH
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if RUN_LOG_PATH is not None:
        return RUN_LOG_PATH

    today_str = datetime.today().strftime("%Y-%m-%d")
    base_name = f"{today_str}_evaluate_and_log"
    ext = ".json"
    index = 0

    while True:
        file_name = f"{base_name}{'' if index == 0 else f'_{index}'}{ext}"
        path = output_dir / file_name
        if not path.exists():
            RUN_LOG_PATH = path
            return path
        index += 1


def evaluate_only_and_log(
    model,
    user_coords,
    evac_candidates,
    mmi_points,
    event_id,
    output_dir="log_eval_and_reward",
):
    from torch import tensor

    global RUN_LOG_PATH

    mmi_coords, mmi_values = prepare_mmi_tensors(mmi_points)
    episode_logs = []
    correct = 0
    found_safe = False

    for user_lat, user_lon in user_coords:
        best_reward = -float("inf")
        best_evac = None
        step_log = []

        for evac_lat, evac_lon in evac_candidates:
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)
            mmi_val, _ = get_nearest_mmi_tensor(
                evac_lat, evac_lon, mmi_coords, mmi_values
            )

            state = tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,
                    mmi_val / 10.0,
                ],
                dtype=torch.float32,
            ).to(DEVICE)

            score = model(state).item()
            reward, mmi_val, dist_km = compute_reward(
                user_lat, user_lon, evac_lat, evac_lon, mmi_coords, mmi_values
            )

            step_log.append(
                {
                    "evac_coord": [evac_lat, evac_lon],
                    "score": score,
                    "reward": reward,
                    "mmi": float(mmi_val),
                    "distance_km": float(dist_km),
                }
            )

            if reward > best_reward:
                best_reward = reward
                best_evac = (evac_lat, evac_lon, mmi_val, dist_km)

        if best_evac is None:
            continue

        evac_lat, evac_lon, mmi_val, dist_km = best_evac
        valid_mmi = int(mmi_val < 4.0)
        valid_dist = int(dist_km <= 1.0)
        is_valid = int(valid_mmi and valid_dist)
        correct += is_valid

        if best_reward > 0:
            found_safe = True

        episode_logs.append(
            {
                "user_coord": [user_lat, user_lon],
                "chosen_evac": [evac_lat, evac_lon],
                "mmi": float(mmi_val),
                "reward": float(best_reward),
                "distance_km": float(dist_km),
                "valid_mmi": valid_mmi,
                "valid_distance": valid_dist,
                "valid": is_valid,
                "all_candidates": step_log,
            }
        )

    accuracy = correct / len(user_coords) if user_coords else 0.0

    # ✅ Pakai file log aktif (1x run saja)
    output_path = get_or_create_log_file(output_dir)

    if output_path.exists():
        with open(output_path, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    all_logs[event_id] = {
        "event_id": event_id,
        "success_rate": accuracy,
        "total_users": len(user_coords),
        "successful_routes": int(correct),
        "episodes": episode_logs,
        "no_safe_evac": not found_safe,
    }

    with open(output_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(
        f"{'⚠️' if not found_safe else '✅'} Event '{event_id}' disimpan ke '{output_path.name}' | Success Rate: {accuracy:.3f}"
    )

    return accuracy


# 📁 Simpan dan Muat Cache OSRM
import json


def save_osrm_cache(path="osrm_cache.json"):
    with open(path, "w") as f:
        json.dump(osrm_cache, f, indent=2)
    print(f"💾 OSRM cache disimpan ke: {path}")


def load_osrm_cache(path="osrm_cache.json"):
    global osrm_cache
    try:
        with open(path, "r") as f:
            osrm_cache = json.load(f)
        print(f"✅ OSRM cache dimuat dari: {path} ({len(osrm_cache)} entri)")
    except FileNotFoundError:
        print("⚠️ File cache tidak ditemukan, mulai dari kosong.")
