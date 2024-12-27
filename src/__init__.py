import os

# Define the base data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

# Define subdirectories
MODEL_FILES_DIR = os.path.join(DATA_DIR, "model_files")
REPRESENTATIONS_DIR = os.path.join(DATA_DIR, "representations")
TRAJECTORIES_DIR = os.path.join(DATA_DIR, "trajectories")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
JOINED_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "joined")

# Ensure the directories exist
os.makedirs(MODEL_FILES_DIR, exist_ok=True)
os.makedirs(REPRESENTATIONS_DIR, exist_ok=True)
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(JOINED_VIDEOS_DIR, exist_ok=True)
