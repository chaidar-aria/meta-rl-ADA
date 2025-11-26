import csv
import json
import os 

#  Nama file CSV 
FILE_CSV = 'titik_evakuasi_2_dengan_alamat baru.csv' 

#  Membuat nama file output GeoJSON secara otomatis
# (Disimpan di folder yang sama dengan file Python)
nama_file_tanpa_ekstensi = os.path.splitext(os.path.basename(FILE_CSV))[0]
FILE_GEOJSON = nama_file_tanpa_ekstensi + '.geojson'

# Menyesuaikan dengan nama kolom ASLI di file CSV 
KOLOM_LAT = 'kor_lat'
KOLOM_LON = 'kor_lon'

# Struktur dasar GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": []
}

print(f"Membaca file CSV dari: {FILE_CSV}")

# Membaca CSV
try:
    with open(FILE_CSV, mode='r', encoding='utf-8-sig') as f_csv:
        # Membaca CSV sebagai dictionary
        reader = csv.DictReader(f_csv)

        for baris in reader:
            try:
                # Mengambil data lat/lon dan ubah ke angka (float)
                lat = float(baris[KOLOM_LAT]) # Mengambil dari 'kor_lat'
                lon = float(baris[KOLOM_LON]) # Mengambil dari 'kor_lon'

                # Salin semua data dari baris CSV ke 'properties'
                properti = baris.copy()
                
                # Menghapus data lat/lon dari properties agar tidak duplikat
                properti.pop(KOLOM_LAT, None) 
                properti.pop(KOLOM_LON, None)

                # Membuat struktur 'Feature'
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        # Standar GeoJSON: [longitude, latitude]
                        "coordinates": [lon, lat] 
                    },
                    "properties": properti
                }
                
                geojson["features"].append(feature)

            except (ValueError, KeyError, TypeError):
                print(f"Melewatkan baris karena data lat/lon tidak valid: {baris}")
            except KeyError:
                print(f"Error: Kolom tidak ditemukan. Pastikan nama '{KOLOM_LAT}' dan '{KOLOM_LON}' ada di CSV.")
                break # Hentikan jika nama kolom salah

    # Menulis hasil ke file GeoJSON
    with open(FILE_GEOJSON, 'w', encoding='utf-8') as f_geojson:
        json.dump(geojson, f_geojson, indent=2)

    # Mendapatkan path absolut (lengkap) dari file yang baru dibuat
    lokasi_file_absolut = os.path.abspath(FILE_GEOJSON)

    print("-" * 30) 
    print(f"Selesai! File '{os.path.basename(FILE_GEOJSON)}' telah dibuat.")
    print(f"Lokasi file tersimpan di: {lokasi_file_absolut}")
    print("-" * 30)

except FileNotFoundError:
    print(f"ERROR: File CSV tidak ditemukan.")
    print(f"Pastikan file '{FILE_CSV}' berada di folder yang sama dengan skrip Python Anda.")
except Exception as e:
    print(f"Terjadi error: {e}")