from fastapi import APIRouter, UploadFile, File
from app.schemas.upload import FileUploadResponse
from app.services.upload_service import upload_files as svc_upload_files

router = APIRouter()

@router.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: list[UploadFile] = File(...), client_id: str = "default"):
    return await svc_upload_files(files, client_id)
