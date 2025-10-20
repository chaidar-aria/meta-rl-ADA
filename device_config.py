# device_config.py
# Mengelola pengaturan perangkat (CPU/GPU) untuk PyTorch.

import torch

# Tentukan perangkat yang akan digunakan (GPU jika tersedia, jika tidak CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def info_device():
    """Mencetak informasi tentang perangkat yang digunakan."""
    print(f"=========================================")
    if DEVICE.type == "cuda":
        print(f"✅ Perangkat aktif: {torch.cuda.get_device_name(0)}")
    else:
        print(f"✅ Perangkat aktif: CPU")
    print(f"=========================================")
