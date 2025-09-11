# device_config.py
import torch
import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def info_device(log_file="device_log.txt"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        msg = f"[{now}] ✅ GPU aktif: {device_name}"
    else:
        msg = f"[{now}] ⚠️ CPU digunakan. (GPU tidak tersedia)"

    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
