from pydantic import BaseModel


class MindmapGenerateRequest(BaseModel):
    dataset_name: str
    materialize: bool = False


class MindmapGenerateResponse(BaseModel):
    success: bool
    message: str
    dataset_name: str
    mindmap: dict
    save_path: str | None = None


class MindmapMaterializeRequest(BaseModel):
    dataset_name: str


class MindmapQARequest(BaseModel):
    dataset_name: str
    save: bool = True
    limit_per_module: int = 10


class MindmapQAResponse(BaseModel):
    success: bool
    saved_files: list
    stats: dict


class MindmapQAMdRequest(BaseModel):
    dataset_name: str
    src: str
    filename: str
    top_k_triples: int = 12
    top_k_chunks: int = 6
    prompt_strategy: str = "auto"


class MindmapQAMdResponse(BaseModel):
    success: bool
    saved_files: list
    stats: dict

