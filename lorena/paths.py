"""Caminhos absolutos: o app funciona independente do diretório de execução."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

ASSETS_DIR = PACKAGE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "imgs"
MODELS_DIR = ASSETS_DIR / "models"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
LOGO_IMAGE = IMAGES_DIR / "logo.png"

SETTINGS_DIR = PROJECT_DIR / "settings"
CONFIG_FILE = SETTINGS_DIR / "config.json"
CALIBRATION_DIR = SETTINGS_DIR / "calibration"
LOG_DIR = PROJECT_DIR / "logs"
