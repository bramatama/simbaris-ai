import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "result")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Cascade classifier
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Algorithm parameters
MIN_AREA = 20000
MAX_AREA = 150000
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85
TEXT_SEARCH_HEIGHT = 70
TEXT_SEARCH_WIDTH_TOLERANCE = 40
EASYOCR_CONFIDENCE_THRESHOLD = 0.3

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)