# logging_utils.py
# Mengelola fungsi-fungsi untuk menyimpan log hasil training dan evaluasi.

import json
from pathlib import Path
from datetime import datetime

# Variabel global untuk path file log dalam satu sesi run
RUN_LOG_PATH = None


def get_or_create_log_file(output_dir="logs"):
    """
    Membuat nama file log unik untuk setiap sesi eksekusi
    dan menyimpannya secara global.
    """
    global RUN_LOG_PATH
    if RUN_LOG_PATH is not None:
        return RUN_LOG_PATH

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.today().strftime("%Y-%m-%d")
    base_name = f"{today_str}_evaluation_log"
    ext = ".json"
    index = 0

    while True:
        file_name = f"{base_name}{'' if index == 0 else f'_{index}'}{ext}"
        path = output_dir / file_name
        if not path.exists():
            RUN_LOG_PATH = path
            return path
        index += 1


def save_multi_event_log(event_id, log_data, log_path):
    """Menyimpan log untuk beberapa event ke dalam satu file JSON."""
    all_logs = {}
    if log_path.exists():
        with open(log_path, "r") as f:
            try:
                all_logs = json.load(f)
            except json.JSONDecodeError:
                pass  # Timpa file jika korup

    all_logs[event_id] = log_data

    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
