GAMMA = 0.9
LEARNING_RATE = 1e-3
MAX_EPISODES = 100
INNER_LR = 1e-2
INNER_STEPS = 5
MAX_META_ITER = 300
EPSILON = 0.1
BASE_URL_OSRM = "http://localhost:3006"

# config.py

# Ganti ini jadi True jika ingin generate titik user dari SHP (evaluasi / inferensi)
USE_SHP_FOR_USER = False

# SHP path jika pakai SHP
SHP_PATH = "./kota-surabaya/PEMUKIMAN_AR_25K.shp"

# Jumlah titik pengguna
N_USER_POINTS = 50

# Path JSON user untuk mode training
USER_JSON_PATH = "./kota-surabaya/user_coords_train.json"
