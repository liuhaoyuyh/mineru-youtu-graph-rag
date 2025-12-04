"""
思维导图服务
"""
from typing import Dict
from fastapi import HTTPException

from utils.logger import logger
from app.utils.mindmap_legacy import (
    generate_mindmap as _generate_mindmap,
    materialize_mindmap as _materialize_mindmap,
    get_mindmap as _get_mindmap,
    get_mindmap_visualization as _get_mindmap_visualization,
    get_mindmap_tree as _get_mindmap_tree,
    mindmap_qa as _mindmap_qa,
    mindmap_qa_md as _mindmap_qa_md,
    MindmapGenerateRequest,
    MindmapMaterializeRequest,
    MindmapQARequest,
    MindmapQAMdRequest,
)


async def generate_mindmap(dataset_name: str, materialize: bool = False) -> Dict:
    """生成思维导图"""
    request = MindmapGenerateRequest(dataset_name=dataset_name, materialize=materialize)
    return await _generate_mindmap(request)


async def materialize_mindmap(dataset_name: str) -> Dict:
    """物化思维导图"""
    request = MindmapMaterializeRequest(dataset_name=dataset_name)
    return await _materialize_mindmap(request)


async def get_mindmap(dataset_name: str) -> Dict:
    """获取思维导图"""
    return await _get_mindmap(dataset_name)


async def get_mindmap_visualization(dataset_name: str) -> Dict:
    """获取思维导图可视化数据"""
    return await _get_mindmap_visualization(dataset_name)


async def get_mindmap_tree(dataset_name: str) -> Dict:
    """获取思维导图树结构"""
    return await _get_mindmap_tree(dataset_name)


async def mindmap_qa(dataset_name: str, save: bool = True, limit_per_module: int = 10, client_id: str = "web_client") -> Dict:
    """思维导图问答"""
    request = MindmapQARequest(dataset_name=dataset_name, save=save, limit_per_module=limit_per_module)
    return await _mindmap_qa(request, client_id)


async def mindmap_qa_md(dataset_name: str, src: str, filename: str, top_k_triples: int = 12, top_k_chunks: int = 6, prompt_strategy: str = "auto", client_id: str = "web_client") -> Dict:
    """思维导图 Markdown 问答"""
    request = MindmapQAMdRequest(
        dataset_name=dataset_name,
        src=src,
        filename=filename,
        top_k_triples=top_k_triples,
        top_k_chunks=top_k_chunks,
        prompt_strategy=prompt_strategy
    )
    return await _mindmap_qa_md(request, client_id)
