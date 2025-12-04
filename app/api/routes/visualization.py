from fastapi import APIRouter
from app.services.mindmap_service import get_mindmap_visualization as svc_get_mindmap_visualization

router = APIRouter()

@router.get("/api/mindmap/visualization/{dataset_name}")
async def get_mindmap_visualization(dataset_name: str):
    return await svc_get_mindmap_visualization(dataset_name)
