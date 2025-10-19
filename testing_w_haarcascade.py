import cv2
import numpy as np
import os
import urllib.request

# --- 1. PENGATURAN ---
# Ganti dengan path lengkap ke file gambar Anda
input_path = "./templates/Template 1.png"
# Ganti dengan nama folder tujuan Anda
output_folder = "./uploads/testing 6/"
# File model Haar Cascade untuk deteksi wajah
cascade_file = "haarcascade_frontalface_default.xml"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


# --- PARAMETER FILTER (Silakan sesuaikan nilai ini) ---
# Filter berdasarkan luas kontur dalam piksel
MIN_AREA = 20000
MAX_AREA = 150000

# Filter berdasarkan rasio aspek (lebar / tinggi)
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85
# --- AKHIR PENGATURAN ---


# --- 2. PERSIAPAN ---
# Membuat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' telah dibuat.")

# Download file Haar Cascade jika belum ada
if not os.path.exists(cascade_file):
    print(f"Downloading {cascade_file}...")
    urllib.request.urlretrieve(cascade_url, cascade_file)
    print("Download selesai.")

# Muat model Haar Cascade
face_cascade = cv2.CascadeClassifier(cascade_file)

# Pastikan file input ada
if not os.path.exists(input_path):
    print(f"Error: File tidak ditemukan di path: {input_path}")
else:
    # --- 3. PROSES GAMBAR ---
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 4. DETEKSI DAN FILTER KONTUR ---
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / h
        
        if MIN_AREA < area < MAX_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            detected_boxes.append((x, y, w, h))

    detected_boxes = sorted(detected_boxes, key=lambda b: (b[1], b[0]))

    # --- 5. CROP, VALIDASI WAJAH, DAN SIMPAN ---
    count_saved = 0
    for i, (x, y, w, h) in enumerate(detected_boxes):
        # Crop gambar dari image asli
        crop = image[y:y+h, x:x+w]
        # Ubah ke grayscale untuk deteksi wajah
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah di dalam area crop
        faces = face_cascade.detectMultiScale(crop_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # HANYA simpan jika wajah terdeteksi
        if len(faces) > 0:
            output_path = os.path.join(output_folder, f"foto_{count_saved}.png")
            cv2.imwrite(output_path, crop)
            count_saved += 1

    print("-" * 30)
    if count_saved > 0:
        print(f"Selesai! {count_saved} foto yang mengandung wajah berhasil divalidasi dan disimpan di '{output_folder}'.")
    else:
        print("Tidak ada foto yang mengandung wajah terdeteksi dengan kriteria filter yang ditentukan.")

