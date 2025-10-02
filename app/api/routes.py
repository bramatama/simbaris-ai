from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.contour_service import detect_contours

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    coords = await detect_contours(file)
    return JSONResponse(content={"detected_boxes": coords})
