-----

# 📘 README.md - Smart Flood Evacuation System (RL)

## 🌟 Deskripsi Proyek

Proyek ini menggunakan **Reinforcement Learning (RL)** untuk mencari rute evakuasi banjir yang optimal. Agen cerdas dilatih untuk mengevakuasi warga dari titik pemukiman ke titik kumpul aman dengan mempertimbangkan tiga faktor utama (Multi-Objective Optimization):

1.  **⛔ Keamanan Banjir:** Menghindari area yang tergenang (Wajib).
2.  **⛰️ Elevasi (Ketinggian):** Memprioritaskan lokasi yang lebih tinggi untuk keamanan jangka panjang.
3.  **🚀 Efisiensi Jarak:** Mencari rute terdekat via jalan raya (menggunakan OSRM).

## 📂 Struktur File (Mandatory)

Pastikan folder proyek Anda memiliki struktur file seperti ini agar kode berjalan lancar:

```text
project_folder/
│
├── kota-surabaya/                     # 📂 FOLDER DATASET
│   ├── Genangan Revisi Lagi.shp       # (Beserta .shx, .dbf, .prj) - Data Banjir
│   ├── PEMUKIMAN_AR_25K.shp           # (Beserta .shx, .dbf, .prj) - Data Sumber Warga
│   ├── titik_evakuasi_2...baru.geojson# Data Titik Kumpul (Format GeoJSON)
│   └── output_hh.tif                  # Data Elevasi (DEM/GeoTIFF)
│
├── config.py                          # ⚙️ Konfigurasi Path & Parameter Training
├── data_utils.py                      # 🛠️ Fungsi baca TIF, GeoJSON, & SHP
├── rl_core.py                         # 🧠 Otak Agen (Logika Reward & Rollout)
├── flood_trainer.py                   # 🏋️‍♂️ Loop Training (Episode & Update Bobot)
├── model.py                           # 🕸️ Arsitektur Neural Network (PyTorch)
├── osrm_utils.py                      # 🛣️ Penghitung Jarak (API OSRM)
├── device_config.py                   # 🖥️ Deteksi CPU/GPU
├── main_training.ipynb                # 📓 JUPYTER NOTEBOOK (Tempat Eksekusi Utama)
│
└── osrm_cache.json                    # (Otomatis dibuat) Cache jarak agar training cepat
```

-----

## 🚀 Panduan Instalasi & Menjalankan

### 1\. Prasyarat (Requirements)

Pastikan Python sudah terinstall. Install library yang dibutuhkan:

```bash
pip install torch numpy geopandas rasterio shapely matplotlib requests contextily
```

*(Catatan: Pastikan OSRM server lokal sudah berjalan, atau sesuaikan `BASE_URL_OSRM` di `config.py` jika menggunakan server publik/lokal).*

### 2\. Cara Menjalankan

Sistem ini dirancang secara **Modular**. Logika ada di file `.py`, tapi eksekusi dilakukan di **Jupyter Notebook**.

1.  Buka **VS Code** atau **Jupyter Lab**.
2.  Buka file `main_training.ipynb`.
3.  Jalankan cell secara berurutan (Run All).
4.  Tunggu proses training selesai.
5.  Hasil model akan disimpan sebagai `.pt` dan grafik performa akan muncul.

-----

# 🛠️ Panduan Modifikasi (User Guide)

Bagian ini menjelaskan file mana yang harus Anda ubah jika ingin mengganti data, mengubah logika, atau menyetel parameter.

### A. Saya ingin ganti Dataset (Peta Baru)

Buka file: **`config.py`**
Di sini Anda mengubah lokasi file peta jika Anda pindah kota atau update data.

```python
# Ubah path di bawah ini sesuai nama file baru Anda
FLOOD_SHP_PATH = "./kota-baru/data_banjir_2025.shp"
EVAC_GEOJSON_PATH = "./kota-baru/titik_kumpul.geojson"
ELEVATION_TIF_PATH = "./kota-baru/demnum_nasional.tif"
```

### B. Saya ingin agen lebih "Galak" soal Jarak (Logika Reward)

Buka file: **`rl_core.py`**
Cari fungsi `compute_multi_objective_reward`. Di sini Anda bisa mengatur bobot poin.

  * **Kasus:** Agen terlalu malas, mau lari ke tempat jauh asalkan tinggi.
  * **Solusi:** Perbesar hukuman jarak atau perkecil bonus elevasi.

<!-- end list -->

```python
# Contoh Modifikasi di rl_core.py

# PERPERAH HUKUMAN JARAK
# Lama: reward -= (dist_km * 15.0)
# Baru (Biar agen gak mau lari jauh sama sekali):
reward -= (dist_km * 50.0) 

# BATASI BONUS KETINGGIAN
# Lama: reward += min(bonus, 30.0)
# Baru (Biar agen gak terlalu tergiur tempat tinggi):
reward += min(bonus, 10.0) 
```

### C. Saya ingin Training lebih lama/cepat

Buka file: **`config.py`** (atau cell konfigurasi di Notebook).

```python
# Perbanyak episode biar makin pintar (tapi lama)
NUM_EPISODES = 500 

# Percepat belajar (Learning Rate)
# Hati-hati, kalau terlalu besar (misal 0.1) agen jadi tidak stabil
LEARNING_RATE = 1e-3 
```

### D. Saya ingin melihat Nama Tempat di Log

Buka file: **`data_loader.py`** atau **`data_utils.py`** (tergantung versi terakhir).
Pastikan fungsi `load_evac_candidates` membaca kolom yang benar dari GeoJSON.

```python
# Cek nama kolom di file GeoJSON Anda (misal namanya "namalokasi", bukan "Evakuasi")
# Ubah baris ini:
name = row.get("namalokasi", f"Titik_{row.name}") 
```

### E. Model tidak mau belajar (Reward stagnan/turun)

1.  **Cek Data Elevasi:** Pastikan file TIF mencakup area koordinat User dan Evakuasi. Jika di luar area, nilai elevasi jadi 0, agen bingung.
2.  **Cek OSRM:** Jika internet mati atau server OSRM mati, jarak mungkin terbaca 0 atau infinity.
3.  **Reset Cache:** Hapus file `osrm_cache.json` agar jarak dihitung ulang.

-----

### 📝 Cheat Sheet File (Ringkasan)

| Ingin Mengubah... | Edit File Ini |
| :--- | :--- |
| **Lokasi File / Path** | `config.py` |
| **Jumlah Episode Training** | `config.py` |
| **Logika Poin (Reward)** | `rl_core.py` |
| **Cara Baca Data (SHP/TIF)** | `data_utils.py` |
| **Arsitektur AI (Layer/Neuron)**| `model.py` |
| **Format Log Tampilan** | `flood_trainer.py` |
| **Eksekusi Program** | `main_training.ipynb` |