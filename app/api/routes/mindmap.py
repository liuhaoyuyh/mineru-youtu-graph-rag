from fastapi import APIRouter
from app.services.mindmap_service import (
    generate_mindmap as svc_generate_mindmap,
    materialize_mindmap as svc_materialize_mindmap,
    get_mindmap as svc_get_mindmap,
    get_mindmap_tree as svc_get_mindmap_tree,
    mindmap_qa as svc_mindmap_qa,
    mindmap_qa_md as svc_mindmap_qa_md,
)
from app.schemas.mindmap import (
    MindmapGenerateRequest,
    MindmapGenerateResponse,
    MindmapMaterializeRequest,
    MindmapQARequest,
    MindmapQAResponse,
    MindmapQAMdRequest,
    MindmapQAMdResponse,
)

router = APIRouter()


@router.post("/api/mindmap/generate", response_model=MindmapGenerateResponse)
async def generate_mindmap(request: MindmapGenerateRequest):
    return await svc_generate_mindmap(request.dataset_name, request.materialize)


@router.post("/api/mindmap/materialize")
async def materialize_mindmap(request: MindmapMaterializeRequest):
    return await svc_materialize_mindmap(request.dataset_name)


@router.get("/api/mindmap/{dataset_name}")
async def get_mindmap(dataset_name: str):
    return await svc_get_mindmap(dataset_name)


@router.get("/api/mindmap-tree/{dataset_name}")
async def get_mindmap_tree(dataset_name: str):
    return await svc_get_mindmap_tree(dataset_name)


@router.post("/api/mindmap/qa", response_model=MindmapQAResponse)
async def mindmap_qa(request: MindmapQARequest, client_id: str = "web_client"):
    return await svc_mindmap_qa(request.dataset_name, request.save, request.limit_per_module, client_id)


@router.post("/api/mindmap/qa/md", response_model=MindmapQAMdResponse)
async def mindmap_qa_md(request: MindmapQAMdRequest, client_id: str = "web_client"):
    return await svc_mindmap_qa_md(request.dataset_name, request.src, request.filename, request.top_k_triples, request.top_k_chunks, request.prompt_strategy, client_id)
