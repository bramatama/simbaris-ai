import cv2
import numpy as np
import os
import shutil

async def detect_contours(file):
    temp_path = f"uploads/{file.filename}"

    # simpan file upload sementara
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # baca gambar
    img = cv2.imread(temp_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # cari kontur
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            coords.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    os.remove(temp_path)  # hapus file sementara
    return coords
