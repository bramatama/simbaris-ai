import cv2
import numpy as np
import os

# --- 1. PENGATURAN ---
# Ganti dengan path lengkap ke file gambar Anda
input_path = "./templates/Template 3.png"
# Ganti dengan nama folder tujuan Anda
output_folder = "./uploads/testing 5/"

# --- PARAMETER FILTER (Silakan sesuaikan nilai ini) ---
# Filter berdasarkan luas kontur dalam piksel
MIN_AREA = 20000  # Luas area minimal yang dianggap sebagai foto
MAX_AREA = 150000 # Luas area maksimal untuk menghindari objek besar yang bukan foto

# Filter berdasarkan rasio aspek (lebar / tinggi)
# Pas foto 3x4 memiliki rasio aspek 0.75. Kita beri rentang toleransi.
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85
# --- AKHIR PENGATURAN ---


# --- 2. PERSIAPAN ---
# Membuat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' telah dibuat.")

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
        # Dapatkan bounding box untuk menghitung properti kontur
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Hitung luas area dan rasio aspek
        area = w * h
        aspect_ratio = float(w) / h
        
        # Terapkan filter
        if MIN_AREA < area < MAX_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            detected_boxes.append((x, y, w, h))

    # Urutkan berdasarkan posisi (atas-bawah, lalu kiri-kanan)
    detected_boxes = sorted(detected_boxes, key=lambda b: (b[1], b[0]))

    # --- 5. CROP DAN SIMPAN GAMBAR ---
    count_saved = 0
    for i, (x, y, w, h) in enumerate(detected_boxes):
        # Crop gambar dari image asli
        crop = image[y:y+h, x:x+w]
        
        # Buat nama file untuk setiap hasil crop (e.g., foto_0.png, foto_1.png)
        output_path = os.path.join(output_folder, f"foto_{i}.png")
        
        # Simpan file hasil crop
        cv2.imwrite(output_path, crop)
        count_saved += 1

    print("-" * 30)
    if count_saved > 0:
        print(f"Selesai! {count_saved} foto berhasil dideteksi dan disimpan di '{output_folder}'.")
    else:
        print("Tidak ada foto yang terdeteksi dengan kriteria filter yang ditentukan.")
