from fastapi import FastAPI
from app.api.routes import router as api_router
from app.api.photo_routes import router as photo_router

app = FastAPI(title="Contour Detection Service")

# Register API routes
app.include_router(api_router, prefix="/api")
app.include_router(photo_router, prefix="/api/photos", tags=["photos"])
