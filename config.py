# config.py
import os

# Base Directory
DATA_DIR = "./dataset"

# File Paths (SESUAIKAN DENGAN FILE BARU)
FLOOD_SHP_PATH = os.path.join(DATA_DIR, "Genangan Revisi Lagi.shp")
EVAC_GEOJSON_PATH = os.path.join(
    DATA_DIR, "titik_evakuasi_2_dengan_alamat baru.geojson"
)
USER_SHP_PATH = os.path.join(DATA_DIR, "PEMUKIMAN_AR_25K.shp")
ELEVATION_TIF_PATH = os.path.join(DATA_DIR, "output_hh.tif")

# Hyperparameters
LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 200


# OSRM Server Configuration
BASE_URL_OSRM = "http://localhost:3006"
