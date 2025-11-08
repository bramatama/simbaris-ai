import cv2
import numpy as np
import os
import urllib.request
import re
import easyocr
import json


NAMA_FILE = "13+1.jpg"
INPUT_PATH = f"./templates/{NAMA_FILE}"
OUTPUT_FOLDER = f"./result_2.0/{NAMA_FILE}/"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

MIN_AREA = 20000
MAX_AREA = 150000
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85
TEXT_SEARCH_HEIGHT = 70 
TEXT_SEARCH_WIDTH_TOLERANCE = 40
EASYOCR_CONFIDENCE_THRESHOLD = 0.3 # Ambang batas kepercayaan untuk EasyOCR (0-1)

# 2. FUNGSI-FUNGSI BANTUAN (HELPER FUNCTIONS)

def sanitize_filename(name):
    """Membersihkan string untuk dijadikan nama file yang valid."""
    name = re.sub(r'[_.:]', ' ', name)
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def prepare_environment(OUTPUT_FOLDER):
    """Membuat folder output dan mengunduh file cascade jika diperlukan."""
    # Normalisasi dan pastikan path diakhiri separator
    OUTPUT_FOLDER = os.path.normpath(OUTPUT_FOLDER) + os.sep

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Folder '{OUTPUT_FOLDER}' telah dibuat.")
    else:
        base = OUTPUT_FOLDER.rstrip('/\\')
        i = 1
        # Cari nama folder baru yang belum ada: base(1), base(2), ...
        while True:
            candidate = f"{base}({i}){os.sep}"
            if not os.path.exists(candidate):
                os.makedirs(candidate)
                OUTPUT_FOLDER = candidate
                print(f"Folder '{base}{os.sep}' sudah ada. Menggunakan folder baru '{OUTPUT_FOLDER}'.")
                break
            i += 1
    
    if not os.path.exists(CASCADE_FILE):
        print(f"Downloading {CASCADE_FILE}...")
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)
        print("Download selesai.")
    return OUTPUT_FOLDER

# =============================================================================
# 3. FUNGSI-FUNGSI UTAMA (CORE FUNCTIONS)
# =============================================================================

def detect_all_text_globally(image_gray, reader):
    """Mendeteksi semua kata dan lokasinya di seluruh gambar menggunakan EasyOCR."""
    print("Mendeteksi semua teks di dokumen dengan EasyOCR...")
    # EasyOCR.readtext mengembalikan list dari (bbox, text, confidence)
    text_data = reader.readtext(image_gray)
    
    all_words = []
    for (bbox, text, conf) in text_data:
        if conf > EASYOCR_CONFIDENCE_THRESHOLD and text.strip():
            # bbox adalah [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (top-left, top-right, bottom-right, bottom-left)
            # Kita ubah ke format [x, y, w, h] yang diharapkan
            (tl, tr, br, bl) = bbox
            x = int(tl[0])
            y = int(tl[1])
            w = int(br[0] - tl[0])
            h = int(br[1] - tl[1])

            word = {
                'text': text,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            }
            all_words.append(word)
    
    print(f"Total {len(all_words)} blok teks terdeteksi.")
    return all_words

def detect_photo_contours(image_gray):
    """Mendeteksi semua kontur yang berpotensi sebagai foto."""
    print("Mendeteksi kontur foto...")
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
            
    detected_boxes = sorted(detected_boxes, key=lambda b: (b[1], b[0]))
    print(f"Total {len(detected_boxes)} kontur foto terdeteksi.")
    return detected_boxes

def match_text_to_photo(photo_box_coords, all_words):
    """Mencari dan menggabungkan kata-kata yang cocok untuk sebuah foto."""
    x, y, w, h = photo_box_coords
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
    clean_full_name = sanitize_filename(raw_full_name)

    bounding_box_data = {"full_name": clean_full_name, "words": candidate_words}
    
    return clean_full_name, bounding_box_data

# =============================================================================
# 4. ALUR KERJA UTAMA (MAIN WORKFLOW)
# =============================================================================

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline."""
    final_output_folder = prepare_environment(OUTPUT_FOLDER)
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    
    # Inisialisasi EasyOCR Reader (cukup sekali)
    # Kita gunakan bahasa Indonesia ('id') dan Inggris ('en')
    print("Memuat model EasyOCR (mungkin perlu waktu saat pertama kali)...")
    reader = easyocr.Reader(['id', 'en']) 
    print("Model EasyOCR dimuat.")

    if not os.path.exists(INPUT_PATH):
        print(f"Error: File tidak ditemukan di path: {INPUT_PATH}")
        return

    image = cv2.imread(INPUT_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Langkah 1: Deteksi semua teks di seluruh dokumen
    # Kirim gambar grayscale dan reader yang sudah diinisialisasi
    all_words = detect_all_text_globally(gray, reader)
    
    # Langkah 2: Deteksi semua kontur foto
    photo_boxes = detect_photo_contours(gray)

    # Langkah 3: Proses setiap foto
    count_saved = 0
    for i, (x, y, w, h) in enumerate(photo_boxes):
        crop_foto = image[y:y+h, x:x+w]
        
        # Validasi Wajah
        faces = face_cascade.detectMultiScale(crop_foto, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Cocokkan dengan Teks
            nama, bounding_box_data = match_text_to_photo((x, y, w, h), all_words)
            
            filename_base = nama if nama else f"tanpa_nama_{count_saved}" 

            # Simpan Hasil
            image_output_path = os.path.join(final_output_folder, f"{filename_base}.png")
            json_output_path = os.path.join(final_output_folder, f"{filename_base}.json")
            
            cv2.imwrite(image_output_path, crop_foto)
            if bounding_box_data:
                with open(json_output_path, 'w') as f:
                    json.dump(bounding_box_data, f, indent=4)
            
            count_saved += 1
            print(f"-> Foto terdeteksi: '{nama}' -> Disimpan sebagai '{filename_base}.png'")

    print("-" * 30)
    if count_saved > 0:
        print(f"Selesai! {count_saved} foto berhasil diproses dan disimpan di '{final_output_folder}'.")
    else:
        print("Tidak ada foto yang memenuhi semua kriteria (bentuk, wajah, dan teks).")

if __name__ == "__main__":
    main()
