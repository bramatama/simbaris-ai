# Photo and Text Detection Service

This service processes document images to detect photos, recognize faces, and extract associated text (names) using computer vision and OCR technologies. It provides both a command-line interface and a REST API.

## Features

- Photo detection using contour analysis
- Face detection using Haar Cascade Classifier
- Text extraction using two OCR options:
  - Tesseract OCR
  - EasyOCR (recommended)
- Automatic text-to-photo matching
- JSON metadata generation
- REST API for remote processing

## Prerequisites

- Python 3.8+
- OpenCV
- EasyOCR or Tesseract OCR
- FastAPI (for API service)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bramatama/simbaris-ai.git
cd simbaris-ai
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Make sure you have Tesseract installed if you plan to use the Tesseract OCR version:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`

## Usage

### Command Line Interface

There are three main script variants:

1. Basic Contour Detection:

```bash
python testing.py
```

2. Tesseract OCR Version:

```bash
python testing_w_tesseract.py
```

3. EasyOCR Version (Recommended):

```bash
python fix_algorithm.py
```

### REST API

1. Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

2. Access the API:

   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. Process photos via API:

```bash
curl -X POST "http://localhost:8000/api/photos/process-photos" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

## API Endpoints

### POST /api/photos/process-photos

Process an image to detect photos and associated text.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: form field "file" with image

**Response:**

```json
{
  "success": true,
  "output_folder": "result/processed_image(1)",
  "total_processed": 2,
  "results": [
    {
      "name": "John Doe",
      "image_path": "result/processed_image(1)/john_doe.png",
      "json_path": "result/processed_image(1)/john_doe.json",
      "bbox": {
        "x": 100,
        "y": 200,
        "w": 300,
        "h": 400
      }
    }
  ]
}
```

## Configuration

Key parameters can be adjusted in `app/config.py`:

```python
MIN_AREA = 20000
MAX_AREA = 150000
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 0.85
TEXT_SEARCH_HEIGHT = 70
TEXT_SEARCH_WIDTH_TOLERANCE = 40
EASYOCR_CONFIDENCE_THRESHOLD = 0.3
```

## Project Structure

```
├── app/
│   ├── api/
│   │   ├── photo_routes.py
│   │   └── routes.py
│   ├── services/
│   │   ├── contour_processing.py
│   │   └── contour_service.py
│   ├── config.py
│   └── main.py
├── result/
├── templates/
├── uploads/
├── fix_algorithm.py
├── testing_w_tesseract.py
├── testing_w_haarcascade.py
├── testing.py
└── requirements.txt
```

## Output Format

### Image Files

- Detected photos are saved as PNG files
- Filenames are sanitized versions of detected names
- If no name is detected, files are named "tanpa_nama_N"

### JSON Metadata

For each detected photo, a JSON file is created containing:

- Full name
- Word positions and bounding boxes
- Confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Authors

- [bramatama](https://github.com/bramatama)

## Acknowledgments

- OpenCV for computer vision capabilities
- EasyOCR and Tesseract for OCR functionality
- FastAPI for the web framework
