

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Membaca file CSV 
input_file = "e:/IBAD 21/RISET/coding/fix/evakuasi/titik_evakuasi_2_fix.csv"
df = pd.read_csv(input_file)

# Meyiapkan geolocator 
geolocator = Nominatim(user_agent="evakuasi_geocoder")

# Membatasi agar tidak melanggar rate limit (1 permintaan/detik)
reverse_id = RateLimiter(lambda loc: geolocator.reverse(loc, language="id"), min_delay_seconds=1)
reverse_en = RateLimiter(lambda loc: geolocator.reverse(loc, language="en"), min_delay_seconds=1)

# Fungsi bantu
def get_address(lat, lon, lang="id"):
    try:
        if lang == "id":
            location = reverse_id((lat, lon))
        else:
            location = reverse_en((lat, lon))
        return location.address if location else None
    except Exception as e:
        print(f"⚠️ Gagal pada titik ({lat}, {lon}): {e}")
        return None

# Menambahkan kolom alamat 
alamat_id_list = []
alamat_en_list = []

print("🔄 Memproses semua titik...")

for idx, row in df.iterrows():
    lat, lon = row["kor_lat"], row["kor_lon"]
    print(f"📍 [{idx+1}/{len(df)}] Memproses koordinat ({lat}, {lon}) ...")
    
    alamat_id = get_address(lat, lon, "id")
    alamat_en = get_address(lat, lon, "en")

    alamat_id_list.append(alamat_id)
    alamat_en_list.append(alamat_en)

    # Delay tambahan aman
    time.sleep(1)

df["Alamat_ID"] = alamat_id_list
df["Alamat_EN"] = alamat_en_list

# Menyimpan hasil 
output_file = "e:/IBAD 21/RISET/coding/fix/evakuasi/titik_evakuasi_2_dengan_alamat baru.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n Proses selesai")
print(f"Hasil disimpan di: {output_file}")
