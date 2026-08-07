import gdown
import os

# Anchor the path to this script's own folder, regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "species_id", "vedavision_species_model.pkl")
FILE_ID = "1pjcLo9X3P80LMo-e0BAFpbtYXcbLKYIJ"  # your actual ID

def ensure_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already present at {MODEL_PATH}, skipping download.")
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading model to {MODEL_PATH}...")
    gdown.download(url, MODEL_PATH, quiet=False)

if __name__ == "__main__":
    ensure_model()