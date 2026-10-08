from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datasets'
OUTPUT_DIR = ROOT / 'outputs'
MODELS_DIR = ROOT / 'models'
DATASET_PATH = DATA_DIR / '1429_1.csv'

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
