from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import os
import urllib.request
import re
import easyocr
import json
from pathlib import Path
from ..config import *

class ContourProcessingService:
    def __init__(self):
        self._ensure_cascade_file()
        self.face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['id', 'en'])
        print("EasyOCR initialized.")
    
    def _ensure_cascade_file(self):
        """Ensure the cascade classifier file exists."""
        if not os.path.exists(CASCADE_FILE):
            print(f"Downloading {CASCADE_FILE}...")
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)
            print("Download complete.")

    def _sanitize_filename(self, name: str) -> str:
        """Clean string for valid filename."""
        name = re.sub(r'[_.:]', ' ', name)
        name = re.sub(r'[^a-zA-Z\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _create_output_folder(self, base_name: str) -> str:
        """Create and return output folder path."""
        output_folder = os.path.join(RESULT_DIR, base_name)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            return output_folder
        
        counter = 1
        while True:
            new_folder = f"{output_folder}({counter})"
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
                return new_folder
            counter += 1

    def detect_all_text(self, image_gray: np.ndarray) -> List[Dict]:
        """Detect text using EasyOCR."""
        print("Detecting text with EasyOCR...")
        text_data = self.reader.readtext(image_gray)
        
        all_words = []
        for (bbox, text, conf) in text_data:
            if conf > EASYOCR_CONFIDENCE_THRESHOLD and text.strip():
                (tl, tr, br, bl) = bbox
                word = {
                    'text': text,
                    'x': int(tl[0]),
                    'y': int(tl[1]),
                    'w': int(br[0] - tl[0]),
                    'h': int(br[1] - tl[1])
                }
                all_words.append(word)
        
        return all_words

    def detect_photo_contours(self, image_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential photo contours."""
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        _, img_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = float(w) / h
            
            if MIN_AREA < area < MAX_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                detected_boxes.append((x, y, w, h))
                
        return sorted(detected_boxes, key=lambda b: (b[1], b[0]))

    def match_text_to_photo(self, photo_coords: Tuple[int, int, int, int], all_words: List[Dict]) -> Tuple[Optional[str], Optional[Dict]]:
        """Match text to photo based on spatial relationship."""
        x, y, w, h = photo_coords
        photo_bottom = y + h
        
        candidate_words = []
        search_y_max = photo_bottom + TEXT_SEARCH_HEIGHT
        search_x_min = x - TEXT_SEARCH_WIDTH_TOLERANCE
        search_x_max = x + w + TEXT_SEARCH_WIDTH_TOLERANCE

        for word in all_words:
            word_center_x = word['x'] + word['w'] / 2
            is_below = word['y'] > photo_bottom and (word['y'] + word['h']) < search_y_max
            is_aligned = word_center_x > search_x_min and word_center_x < search_x_max
            
            if is_below and is_aligned:
                candidate_words.append(word)
        
        if not candidate_words:
            return None, None

        candidate_words.sort(key=lambda w: w['x'])
        raw_full_name = " ".join([w['text'] for w in candidate_words])
        clean_full_name = self._sanitize_filename(raw_full_name)
        
        return clean_full_name, {"full_name": clean_full_name, "words": candidate_words}

    async def process_image(self, image_path: str, output_base_name: str) -> Dict:
        """Process an image and extract photos with text."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Create output directory
        output_folder = self._create_output_folder(output_base_name)
        
        # Read and process image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect text and photos
        all_words = self.detect_all_text(gray)
        photo_boxes = self.detect_photo_contours(gray)

        results = []
        count_saved = 0
        
        # Process each detected photo
        for i, (x, y, w, h) in enumerate(photo_boxes):
            crop_foto = image[y:y+h, x:x+w]
            faces = self.face_cascade.detectMultiScale(crop_foto, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                nama, bounding_box_data = self.match_text_to_photo((x, y, w, h), all_words)
                filename_base = nama if nama else f"tanpa_nama_{count_saved}"

                # Save results
                image_filename = f"{filename_base}.png"
                json_filename = f"{filename_base}.json"
                
                image_path = os.path.join(output_folder, image_filename)
                json_path = os.path.join(output_folder, json_filename)
                
                cv2.imwrite(image_path, crop_foto)
                if bounding_box_data:
                    with open(json_path, 'w') as f:
                        json.dump(bounding_box_data, f, indent=4)
                
                results.append({
                    "name": nama,
                    "image_path": image_path,
                    "json_path": json_path if bounding_box_data else None,
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })
                count_saved += 1

        return {
            "success": True,
            "output_folder": output_folder,
            "total_processed": count_saved,
            "results": results
        }