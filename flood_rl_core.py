# flood_rl_core.py
# Berisi fungsi-fungsi inti untuk training RL standar pada skenario banjir.

import os
import numpy as np
import torch
from torch.distributions import Categorical

from device_config import DEVICE
from config import GAMMA
from data_utils import get_adaptive_evac_candidates
from osrm_utils import get_osrm_distance_cached
from data_utils import is_coord_in_flood_zone


def compute_flood_reward(user_lat, user_lon, evac_lat, evac_lon, flood_gdf):
    """Menghitung reward berdasarkan status banjir dan jarak."""
    is_flooded = is_coord_in_flood_zone(flood_gdf, evac_lat, evac_lon)
    dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

    if is_flooded:
        return -100.0  # Penalti sangat besar jika titik evakuasi banjir

    # Reward berdasarkan kedekatan jarak jika aman
    if dist_km <= 1.0:
        return 100.0
    if dist_km <= 2.0:
        return 50.0
    if dist_km <= 3.0:
        return 10.0
    return -20.0  # Penalti ringan untuk jarak yang terlalu jauh


def rollout(
    policy,
    user_coords,
    evac_candidates,
    flood_gdf,
    episode_start: int = 1,
    max_steps_per_episode: int = None,
    flood_metric_func=None,
    save_log_path: str = None,
    **get_adaptive_kwargs,
):
    """
    Menjalankan satu 'rollout' (satu pass) untuk semua user_coords.
    Mengembalikan daftar trajectory items dengan log_prob dan reward.

    Args:
      policy: model/policy callable yang menerima state tensor dan mengembalikan score/logit.
      user_coords: iterable of (lat, lon)
      evac_candidates: list of (lat, lon) kandidat evakuasi
      flood_gdf: GeoDataFrame banjir
      episode_start: nomor episode awal (bisa 1)
      max_steps_per_episode: batasi berapa kandidat yang dicoba tiap episode (None = semua)
      flood_metric_func: callable(user_lat, user_lon, evac_lat, evac_lon, **ctx) -> float
                bila diset, nilainya akan dicetak sebagai tambahan (mis. kedalaman genangan)
      save_log_path: jika diset, setiap baris log akan di-append ke file ini.
      get_adaptive_kwargs: diteruskan ke get_adaptive_evac_candidates
    """
    trajectories = []
    # Pastikan file log ada jika path diberikan
    if save_log_path:
        os.makedirs(os.path.dirname(save_log_path) or ".", exist_ok=True)

    for episode_idx, (user_lat, user_lon) in enumerate(
        user_coords, start=episode_start
    ):
        # Dapatkan kandidat evakuasi yang aman saja (menggunakan arg tambahan seperti max_shp_polygons)
        safe_candidates, used_radius = get_adaptive_evac_candidates(
            user_lat,
            user_lon,
            evac_candidates,
            flood_gdf,
            **get_adaptive_kwargs,
        )

        if not safe_candidates:
            line = f"Episode: {episode_idx:03d} | ❌ Tidak ada kandidat aman ditemukan (radius {used_radius or 'N/A'} km)"
            print(line)
            if save_log_path:
                with open(save_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            continue

        # batasi jumlah langkah jika diminta
        candidates_to_try = safe_candidates
        if max_steps_per_episode is not None:
            candidates_to_try = safe_candidates[:max_steps_per_episode]

        episode_trajectories = []
        for step_idx, (evac_lat, evac_lon) in enumerate(candidates_to_try, start=1):
            # Ambil jarak lewat OSRM cache
            dist_km = get_osrm_distance_cached(user_lat, user_lon, evac_lat, evac_lon)

            # Bangun state untuk policy (contoh normalisasi sederhana)
            state = torch.tensor(
                [
                    user_lat / 100.0,
                    user_lon / 100.0,
                    evac_lat / 100.0,
                    evac_lon / 100.0,
                    dist_km / 10.0,
                    0.0,  # status bahaya (sudah difilter aman)
                ],
                dtype=torch.float32,
            ).to(DEVICE)

            # Dapatkan skor dari policy
            try:
                score = policy(state).squeeze()
            except Exception as e:
                # jika policy error, skip langkah ini
                print(
                    f"Warning: policy evaluation gagal pada episode {episode_idx} step {step_idx}: {e}"
                )
                continue

            episode_trajectories.append(
                {
                    "state": state,
                    "score": score,
                    "evac": (evac_lat, evac_lon),
                    "distance": dist_km,
                    "step_idx": step_idx,
                }
            )

        # Jika tidak ada skor valid, lanjut ke episode berikutnya
        if not episode_trajectories:
            continue

        # Gabungkan scores menjadi tensor dan sampling dari distribusi softmax
        scores_tensor = torch.stack([t["score"] for t in episode_trajectories])
        probs = torch.softmax(scores_tensor, dim=0)
        # safety checks
        if torch.any(torch.isnan(probs)) or torch.any(probs <= 0):
            print(
                f"Episode {episode_idx:03d}: warning - probs contain NaN/<=0, skip episode."
            )
            continue

        m = Categorical(probs)
        action = m.sample()
        chosen_idx = int(action.item())
        chosen = episode_trajectories[chosen_idx]
        chosen_lat, chosen_lon = chosen["evac"]
        chosen_dist_km = chosen["distance"]

        # Jika ada fungsi metrik banjir (mis. kedalaman), hitung, tapi jangan paksa
        flood_metric_val = None
        if callable(flood_metric_func):
            try:
                flood_metric_val = float(
                    flood_metric_func(user_lat, user_lon, chosen_lat, chosen_lon)
                )
            except Exception:
                flood_metric_val = None

        # Hitung reward berdasarkan fungsi yang ada
        reward = compute_flood_reward(
            user_lat, user_lon, chosen_lat, chosen_lon, flood_gdf
        )

        # Simpan trajectory (tersedia untuk compute loss nanti)
        trajectories.append(
            {
                "log_prob": m.log_prob(action),
                "reward": reward,
                "episode": episode_idx,
                "step": chosen["step_idx"],
                "evac": (chosen_lat, chosen_lon),
                "flood_metric": flood_metric_val,
                "distance_km": chosen_dist_km,
            }
        )

        # Log ke terminal (tanpa MMI — hanya yang relevan untuk banjir)
        if flood_metric_val is not None:
            log_line = (
                f"Episode: {episode_idx:03d} | Step: {chosen['step_idx']:02d} | "
                f"Aksi: {chosen_lat:.5f},{chosen_lon:.5f} → Reward: {reward:.2f} | "
                f"FloodMetric: {flood_metric_val:.2f} | Jarak: {chosen_dist_km:.2f} km"
            )
        else:
            log_line = (
                f"Episode: {episode_idx:03d} | Step: {chosen['step_idx']:02d} | "
                f"Aksi: {chosen_lat:.5f},{chosen_lon:.5f} → Reward: {reward:.2f} | "
                f"Jarak: {chosen_dist_km:.2f} km"
            )

        print(log_line)
        if save_log_path:
            with open(save_log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

        # Ringkasan singkat episode (rata-rata reward dari langkah yang sudah tersimpan utk episode ini)
        ep_rewards = [t["reward"] for t in trajectories if t["episode"] == episode_idx]
        if ep_rewards:
            avg_reward = float(np.mean(ep_rewards))
            summary_line = f"  🔹 Rata-rata reward episode {episode_idx:03d}: {avg_reward:.2f} ({len(ep_rewards)} langkah)"
            print(summary_line)
            if save_log_path:
                with open(save_log_path, "a", encoding="utf-8") as f:
                    f.write(summary_line + "\n")

    return trajectories


def compute_loss(log_probs, rewards):
    """Menghitung loss REINFORCE."""
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    return -(log_probs * returns).sum()


def loss_from_trajectories(trajectories):
    """
    Utility kecil untuk mengubah hasil dari `rollout` (list of dicts) menjadi
    `compute_loss` input. Mengembalikan loss tensor.
    """
    if not trajectories:
        return None
    log_probs = [t["log_prob"] for t in trajectories]
    rewards = [t["reward"] for t in trajectories]
    return compute_loss(log_probs, rewards)
