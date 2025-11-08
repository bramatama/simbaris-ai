from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from ..services.contour_processing import ContourProcessingService
from ..config import UPLOAD_DIR

router = APIRouter()
processing_service = ContourProcessingService()

@router.post("/process-photos")
async def process_photos(file: UploadFile = File(...)):
    """
    Process an uploaded image to detect photos and associated text.
    
    The endpoint:
    1. Saves the uploaded file temporarily
    2. Processes it using the contour detection algorithm
    3. Returns the processing results
    """
    try:
        # Create a unique filename for the upload
        file_extension = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{Path(file.filename).stem}{file_extension}")
        
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        try:
            results = await processing_service.process_image(
                temp_path,
                f"processed_{Path(file.filename).stem}"
            )
            
            return JSONResponse(content=results)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))