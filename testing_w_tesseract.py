import cv2
import numpy as np
import os
import urllib.request
import re
import pytesseract

input_path = "./templates/Template 1_clearedText.png"

output_folder = "./result/(1_1)hasil_dengan_nama_5/"

cascade_file = "haarcascade_frontalface_default.xml"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- PARAMETER FILTER ---
# Filter Kontur Foto
MIN_AREA = 20000
MAX_AREA = 150000
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85

# Pengaturan Area Pencarian Teks di Bawah Foto
TEXT_SEARCH_HEIGHT = 70 
TEXT_SEARCH_WIDTH_TOLERANCE = 40 # Toleransi bisa diperbesar karena pencocokan lebih cerdas

def sanitize_filename(name):
    """Membersihkan string untuk dijadikan nama file yang valid."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' telah dibuat.")

if not os.path.exists(cascade_file):
    print(f"Downloading {cascade_file}...")
    urllib.request.urlretrieve(cascade_url, cascade_file)
    print("Download selesai.")

face_cascade = cv2.CascadeClassifier(cascade_file)

if not os.path.exists(input_path):
    print(f"Error: File tidak ditemukan di path: {input_path}")
else:
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- LANGKAH BARU: Deteksi semua teks dan lokasinya di seluruh gambar ---
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    all_words = []
    for i in range(len(text_data['text'])):
        raw_conf = text_data['conf'][i]
        raw_text = str(text_data['text'][i]) if text_data['text'][i] is not None else ""
        # Safely parse confidence: pytesseract may return '-1' or non-integer strings
        try:
            conf = int(float(raw_conf))
        except Exception:
            conf = -1

        token = raw_text.strip()
        # Filter: only accept tokens with positive confidence and alphabetic-only text
        # Use str.isalpha() which supports unicode letters; this removes numbers and punctuation
        if conf > 60 and token and all(ch.isalpha() or ch in ".," for ch in token):
            word = {
                'text': token,
                'x': int(text_data['left'][i]),
                'y': int(text_data['top'][i]),
                'w': int(text_data['width'][i]),
                'h': int(text_data['height'][i])
            }
            all_words.append(word)
            print(word)
    
    print(f"Total {len(all_words)} kata terdeteksi di seluruh dokumen.")

    # --- Deteksi Kontur Foto ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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

    # --- Loop utama: Validasi wajah dan cocokkan dengan teks terdekat ---
    count_saved = 0
    for i, (x, y, w, h) in enumerate(detected_boxes):
        crop_foto = image[y:y+h, x:x+w]
        crop_foto_gray = cv2.cvtColor(crop_foto, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(crop_foto_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # --- LOGIKA BARU: Cari teks yang cocok dari daftar teks global ---
            photo_box = {'x': x, 'y': y, 'w': w, 'h': h}
            photo_bottom = photo_box['y'] + photo_box['h']
            
            candidate_words = []
            # Tentukan area pencarian di bawah foto
            search_y_max = photo_bottom + TEXT_SEARCH_HEIGHT
            search_x_min = photo_box['x'] - TEXT_SEARCH_WIDTH_TOLERANCE
            search_x_max = photo_box['x'] + photo_box['w'] + TEXT_SEARCH_WIDTH_TOLERANCE

            for word in all_words:
                word_center_x = word['x'] + word['w'] / 2
                # Cek apakah kata berada di bawah foto dan dalam area pencarian horizontal/vertikal
                is_below = word['y'] > photo_bottom and (word['y'] + word['h']) < search_y_max
                is_aligned = word_center_x > search_x_min and word_center_x < search_x_max
                
                if is_below and is_aligned:
                    candidate_words.append(word)

            nama = ""
            if candidate_words:
                # Urutkan kata-kata kandidat dari kiri ke kanan
                candidate_words.sort(key=lambda w: w['x'])
                # Gabungkan kata-kata menjadi satu string nama
                nama = " ".join([w['text'] for w in candidate_words])
            # --- AKHIR LOGIKA BARU ---

            # Bersihkan nama untuk dijadikan nama file
            filename = sanitize_filename(nama)
            
            # Jika nama kosong, gunakan nama default
            if not filename:
                filename = f"tanpa_nama_{count_saved}"
            
            output_path = os.path.join(output_folder, f"{filename}.png")
            cv2.imwrite(output_path, crop_foto)
            count_saved += 1
            print(f"Foto terdeteksi: '{nama}' -> Disimpan sebagai '{filename}.png'")

    print("-" * 30)
    if count_saved > 0:
        print(f"Selesai! {count_saved} foto berhasil diproses dan disimpan di '{output_folder}'.")
    else:
        print("Tidak ada foto yang memenuhi semua kriteria (bentuk, wajah, dan teks).")

