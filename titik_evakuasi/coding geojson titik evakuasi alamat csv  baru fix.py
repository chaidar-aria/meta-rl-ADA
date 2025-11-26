import csv
import json
import os
import sys

# --- Konfigurasi Anda ---

# 1. Nama file CSV baru Anda
FILE_CSV = 'titik_evakuasi_2_ alamat fix.csv' 

# 2. Membuat nama file output GeoJSON secara otomatis
# (Disimpan di folder yang sama dengan file Python Anda)
nama_file_tanpa_ekstensi = os.path.splitext(os.path.basename(FILE_CSV))[0]
FILE_GEOJSON = nama_file_tanpa_ekstensi + '.geojson'

# 3. Nama kolom (Sudah benar sesuai file Anda)
KOLOM_LAT = 'kor_lat'
KOLOM_LON = 'kor_lon'
# -------------------------

# Cari CSV relatif terhadap lokasi skrip agar tidak error saat dijalankan dari folder lain
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_CSV = os.path.join(SCRIPT_DIR, 'titik_evakuasi_2_ alamat fix.csv')
FILE_GEOJSON = os.path.join(SCRIPT_DIR, 'titik_evakuasi_2_ alamat fix.geojson')

geojson = {"type": "FeatureCollection", "features": []}

print(f"Mencari CSV di: {FILE_CSV}")

try:
    if not os.path.exists(FILE_CSV):
        raise FileNotFoundError(FILE_CSV)

    with open(FILE_CSV, mode='r', encoding='latin-1', newline='') as f_csv:
        reader = csv.DictReader(f_csv)
        print("Header CSV:", reader.fieldnames)

        if not reader.fieldnames:
            print("ERROR: Header CSV tidak terdeteksi.")
            sys.exit(1)

        total = 0
        skipped = 0

        for i, baris in enumerate(reader, start=2):
            total += 1
            lat_raw = (baris.get(KOLOM_LAT) or "").strip()
            lon_raw = (baris.get(KOLOM_LON) or "").strip()

            # normalisasi koma desimal dan spasi
            lat_raw = lat_raw.replace(',', '.')
            lon_raw = lon_raw.replace(',', '.')

            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except Exception:
                print(f"[Baris {i}] Lewatkan: lat/lon tidak valid -> '{lat_raw}', '{lon_raw}'")
                skipped += 1
                continue

            props = baris.copy()
            props.pop(KOLOM_LAT, None)
            props.pop(KOLOM_LON, None)

            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props
            }
            geojson["features"].append(feature)

    # Tulis file GeoJSON
    with open(FILE_GEOJSON, 'w', encoding='utf-8') as f_out:
        json.dump(geojson, f_out, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"Selesai. File: {FILE_GEOJSON}")
    print(f"Baris diproses: {total}, fitur dibuat: {len(geojson['features'])}, dilewatkan: {skipped}")
    print("-" * 30)

except FileNotFoundError:
    print(f"ERROR: File CSV tidak ditemukan di: {FILE_CSV}")
    print(f"Letakkan 'titik_evakuasi_2_ alamat fix.csv' di folder: {SCRIPT_DIR} atau ubah path FILE_CSV.")
except Exception as e:
    print("Terjadi error:", str(e))