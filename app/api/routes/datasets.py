from fastapi import APIRouter, UploadFile, File
from app.services.graph_service import reconstruct_dataset as svc_reconstruct
from app.services.dataset_service import (
    get_datasets as svc_get_datasets,
    upload_schema as svc_upload_schema,
    delete_dataset as svc_delete_dataset,
)

router = APIRouter()

@router.get("/api/datasets")
async def get_datasets():
    return await svc_get_datasets()


@router.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(dataset_name: str, schema_file: UploadFile = File(...)):
    return await svc_upload_schema(dataset_name, schema_file)


@router.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    return await svc_delete_dataset(dataset_name)


@router.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    return await svc_reconstruct(dataset_name, client_id)
