# data_utils.py
# Berisi fungsi untuk mempersiapkan tensor dan memfilter data.

import torch
import numpy as np
from device_config import DEVICE
from osrm_utils import get_osrm_distance_cached


def prepare_mmi_tensors(mmi_points):
    """Mengubah list of dict mmi_points menjadi tensor PyTorch."""
    if not mmi_points:
        return torch.empty(0, 2, dtype=torch.float32).to(DEVICE), torch.empty(
            0, dtype=torch.float32
        ).to(DEVICE)

    coords = torch.tensor(
        [[p["lat"], p["lon"]] for p in mmi_points], dtype=torch.float32
    ).to(DEVICE)
    values = torch.tensor([p["mmi"] for p in mmi_points], dtype=torch.float32).to(
        DEVICE
    )
    return coords, values


def get_nearest_mmi_tensor(lat, lon, mmi_coords, mmi_values):
    """Mencari nilai MMI terdekat untuk sebuah koordinat dari data tensor."""
    if mmi_coords.numel() == 0:
        return torch.tensor(float("inf")).to(DEVICE), torch.tensor(float("inf")).to(
            DEVICE
        )

    user = torch.tensor([lat, lon], dtype=torch.float32).to(DEVICE)
    dists = torch.norm(mmi_coords - user, dim=1)
    idx = torch.argmin(dists)
    # Mengembalikan nilai mmi dan jarak Euclidean (bukan jarak sebenarnya)
    return mmi_values[idx], dists[idx] * 111.0


def get_adaptive_evac_candidates(
    user_lat, user_lon, all_candidates, mmi_coords, mmi_values, max_radius_km=5.0
):
    """
    Secara dinamis mencari kandidat evakuasi dalam radius yang meningkat
    dan menyaring yang tidak aman (MMI >= 4).
    """
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
            # Urutkan berdasarkan jarak terdekat, lalu MMI terendah
            filtered.sort(key=lambda x: (x[2], x[3]))
            coords_only = [(lat, lon) for lat, lon, _, _ in filtered]
            return coords_only, radius

    # Jika tidak ada kandidat yang ditemukan dalam radius maksimal
    return [], None
