#!/usr/bin/env python3
"""
Simple but Complete Youtu-GraphRAG Backend
Integrates real GraphRAG functionality with a simple interface
"""

import os
import re
import sys
import json
import asyncio
import glob
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.logger import logger, setup_logger
import logging
from utils.mineru_adapter import is_mineru_available, parse_with_mineru
import ast

# Try to import GraphRAG components
try:
    from models.constructor import kt_gen as constructor
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
    from config import get_config, ConfigManager
    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"⚠️  GraphRAG components not available: {e}")

app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")

# Mount static files (assets directory)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# Mount frontend directory for frontend assets
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Dict[str, WebSocket] = {}
config = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# Request/Response models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None

class GraphConstructionRequest(BaseModel):
    dataset_name: str
    
class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None

class QuestionRequest(BaseModel):
    question: str
    dataset_name: str

class QuestionResponse(BaseModel):
    answer: str
    sub_questions: List[Dict]
    retrieved_triples: List[str]
    retrieved_chunks: List[str]
    reasoning_steps: List[Dict]
    visualization_data: Dict

class MindmapGenerateRequest(BaseModel):
    dataset_name: str
    materialize: Optional[bool] = False

class MindmapGenerateResponse(BaseModel):
    success: bool
    message: str
    dataset_name: str
    mindmap: Dict
    save_path: Optional[str] = None

class MindmapQARequest(BaseModel):
    dataset_name: str
    save: Optional[bool] = True
    limit_per_module: Optional[int] = 10

class MindmapQAResponse(BaseModel):
    success: bool
    saved_files: List[str]
    stats: Dict

class MindmapQAMdRequest(BaseModel):
    dataset_name: str
    src: str
    filename: str
    top_k_triples: Optional[int] = 12
    top_k_chunks: Optional[int] = 6
    prompt_strategy: Optional[str] = "auto"

class MindmapQAMdResponse(BaseModel):
    success: bool
    saved_files: List[str]
    stats: Dict

def ensure_demo_schema_exists() -> str:
    """Ensure default demo schema exists and return its path."""
    os.makedirs("schemas", exist_ok=True)
    schema_path = "schemas/demo.json"
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person", "location", "organization", "event", "object",
                "concept", "time_period", "creative_work", "biological_entity", "natural_phenomenon"
            ],
            "Relations": [
                "is_a", "part_of", "located_in", "created_by", "used_by", "participates_in",
                "related_to", "belongs_to", "influences", "precedes", "arrives_in", "comparable_to"
            ],
            "Attributes": [
                "name", "date", "size", "type", "description", "status",
                "quantity", "value", "position", "duration", "time"
            ]
        }
        with open(schema_path, 'w') as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path

def get_schema_path_for_dataset(dataset_name: str) -> str:
    """Return dataset-specific schema if present; otherwise fallback to demo schema."""
    if dataset_name and dataset_name != "demo":
        ds_schema = f"schemas/{dataset_name}.json"
        if os.path.exists(ds_schema):
            return ds_schema
    return ensure_demo_schema_exists()

async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    """Send progress update via WebSocket"""
    await manager.send_message({
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, client_id)

async def clear_cache_files(dataset_name: str):
    """Clear all cache files for a dataset before graph construction"""
    try:
        # Clear FAISS cache files
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")
        
        # Clear output chunks
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")
        
        # Clear output graphs
        graph_file = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")
        
        # Clear any other cache files with dataset name pattern
        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*"
        ]
        
        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Cleared cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")
        
        logger.info(f"Cache cleanup completed for dataset: {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")
        # Don't raise exception, just log the error

# Serve frontend HTML
@app.get("/")
async def read_root():
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}

@app.get("/api/status")
async def get_status():
    return {
        "message": "Youtu-GraphRAG Unified Interface is running!", 
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), client_id: str = "default"):
    """Upload files and prepare for graph construction"""
    try:
        # Use original filename (without extension) as dataset name
        # If multiple files, use the first file's name
        main_file = files[0]
        original_name = os.path.splitext(main_file.filename)[0]
        # Clean filename to be filesystem-safe
        dataset_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        dataset_name = dataset_name.replace(' ', '_')
        
        # Add timestamp if dataset already exists
        base_name = dataset_name
        counter = 1
        while os.path.exists(f"data/uploaded/{dataset_name}"):
            dataset_name = f"{base_name}_{counter}"
            counter += 1
            
        upload_dir = f"data/uploaded/{dataset_name}"
        os.makedirs(upload_dir, exist_ok=True)
        
        await send_progress_update(client_id, "upload", 10, "Starting file upload...")
        
        # Process uploaded files
        corpus_data = []
        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process file content
            if file.filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                corpus_data.append({
                    "title": file.filename,
                    "text": content
                })
            elif file.filename.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            corpus_data.extend(data)
                        else:
                            corpus_data.append(data)
                except:
                    # If JSON parsing fails, treat as text
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    corpus_data.append({
                        "title": file.filename,
                        "text": content
                    })
            elif file.filename.lower().endswith((
                '.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'
            )):
                # Try MinerU to parse PDFs/images into text corpus (attempt regardless of availability check)
                try:
                    mineru_out_dir = os.path.join(upload_dir, "mineru_out")
                    logs_dir = "output/logs"
                    os.makedirs(logs_dir, exist_ok=True)
                    try:
                        fh_paths = [getattr(h, 'baseFilename', None) for h in logger.handlers]
                        if not any(p and p.endswith("mineru.log") for p in fh_paths):
                            setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "mineru.log"))
                    except Exception:
                        pass
                    logger.info(f"upload parse start: file='{file_path}' out='{mineru_out_dir}'")
                    parsed_entries = parse_with_mineru(file_path, mineru_out_dir)
                    logger.info(f"upload parse result entries={len(parsed_entries) if parsed_entries else 0}")
                    if parsed_entries:
                        corpus_data.extend(parsed_entries)
                    else:
                        corpus_data.append({"title": file.filename, "text": ""})
                except Exception as e:
                    logger.error(f"MinerU parsing failed for {file.filename}: {e}")
                    corpus_data.append({"title": file.filename, "text": ""})
                    progress = 10 + (i + 1) * 80 // len(files)
                    await send_progress_update(client_id, "upload", progress, f"Processed {file.filename} (MinerU parse error, stored empty)")
                    continue

            progress = 10 + (i + 1) * 80 // len(files)
            await send_progress_update(client_id, "upload", progress, f"Processed {file.filename}")
        
        # Save corpus data（仅保存上传与 MinerU 的结果，不做多模态块合并）
        corpus_path = f"{upload_dir}/corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        # Create dataset configuration
        await create_dataset_config()
        
        await send_progress_update(client_id, "upload", 100, "Upload completed successfully!")
        
        return FileUploadResponse(
            success=True,
            message="Files uploaded successfully",
            dataset_name=dataset_name,
            files_count=len(files)
        )
    
    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_dataset_config():
    """Create dataset configuration"""
    # Ensure default demo schema exists
    ensure_demo_schema_exists()

@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(request: GraphConstructionRequest, client_id: str = "default"):
    """Construct knowledge graph from uploaded data"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name
        
        logs_dir = "output/logs"
        os.makedirs(logs_dir, exist_ok=True)
        try:
            fh_paths = [getattr(h, 'baseFilename', None) for h in logger.handlers]
            if not any(p and p.endswith("construction.log") for p in fh_paths):
                setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "construction.log"))
        except Exception:
            pass
        await send_progress_update(client_id, "construction", 2, "Cleaning old cache files...")
        
        # Clear all cache files before construction
        await clear_cache_files(dataset_name)
        
        await send_progress_update(client_id, "construction", 5, "Initializing graph builder...")
        
        # Get dataset paths
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json" 
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)
        
        if not os.path.exists(corpus_path):
            # Try demo dataset
            corpus_path = "data/demo/demo_corpus.json"
        
        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        await send_progress_update(client_id, "construction", 10, "Loading configuration and corpus...")
        
        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")
        
        # 合并多模态块到语料（构建阶段执行）
        try:
            from utils.multimodal_ingestion import build_chunks_for_dataset
            logger.info(f"multimodal ingestion start: dataset='{dataset_name}'")
            extra_chunks = build_chunks_for_dataset(dataset_name)
            logger.info(f"multimodal ingestion chunks={len(extra_chunks) if extra_chunks else 0}")
            if extra_chunks:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    base_docs = json.load(f)
                merged_docs = (base_docs or []) + extra_chunks
                merged_path = f"data/uploaded/{dataset_name}/merged_corpus.json"
                with open(merged_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_docs, f, ensure_ascii=False, indent=2)
                corpus_path = merged_path
        except Exception as _e:
            logger.warning(f"Multimodal ingestion skipped: {_e}")

        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )
        
        await send_progress_update(client_id, "construction", 20, "Starting entity-relation extraction...")
        
        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Run graph construction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
        
        await send_progress_update(client_id, "construction", 95, "Preparing visualization data...")
        # Load constructed graph for visualization
        graph_path = f"output/graphs/{dataset_name}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)
        
        await send_progress_update(client_id, "construction", 100, "Graph construction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "construction",
                "message": "Graph construction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")
        
        return GraphConstructionResponse(
            success=True,
            message="Knowledge graph constructed successfully",
            graph_data=graph_vis_data
        )
    
    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"Construction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "construction",
                "message": f"Construction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_graph_visualization(graph_path: str) -> Dict:
    """Prepare graph data for visualization"""
    try:
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
        
        # Handle different graph data formats
        if isinstance(graph_data, list):
            # GraphRAG format: list of relationships
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            # Standard format: {nodes: [], edges: []}
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
    
    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}

def _setup_mindmap_logger():
    try:
        logs_dir = "output/logs"
        os.makedirs(logs_dir, exist_ok=True)
        fh_paths = [getattr(h, 'baseFilename', None) for h in logger.handlers]
        if not any(p and p.endswith("mindmap.log") for p in fh_paths):
            setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "mindmap.log"))
    except Exception:
        pass

def find_mineru_source_file(dataset_name: str) -> Optional[str]:
    base = os.path.join("data", "uploaded", dataset_name, "mineru_out")
    cand = os.path.join(base, "compiled_mindmap.md")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(base, dataset_name, "ocr", f"{dataset_name}.md")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(base, dataset_name, "vlm", f"{dataset_name}.md")
    if os.path.isfile(cand):
        return cand
    best = None
    best_len = -1
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".md"):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                        head = fp.read(2048)
                    score = 0
                    if "#" in head:
                        score += 1
                    if "Introduction" in head or "摘要" in head or "Abstract" in head:
                        score += 1
                    if score > best_len:
                        best = p
                        best_len = score
                except Exception:
                    continue
    if best:
        return best
    for root, _, files in os.walk(base):
        for f in files:
            if f == "content-list.json" or f.endswith("content-list.json"):
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    return fp
    return None

def list_mineru_md_files(dataset_name: str) -> List[str]:
    base = os.path.join("data", "uploaded", dataset_name, "mineru_out")
    files = []
    for root, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(".md"):
                files.append(os.path.join(root, f))
    try:
        files.sort(key=lambda p: os.path.basename(p))
    except Exception:
        pass
    return files


from typing import Dict, List, Tuple


def _slugify(name: str) -> str:
    """把标题转成简短 slug，方便前端当 id 用。"""
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^0-9A-Za-z_]+", "", s)
    return s[:64] or "node"


def _extract_section_key(name: str):
    """
    从标题里提取章节号前缀：
    '3 Method'                      -> '3'
    '3.1 Point Cloud Sampling ...'  -> '3.1'
    '4.2 Ablation Study ...'        -> '4.2'
    其它情况返回 None
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\b", name.strip())
    return m.group(1) if m else None


def parse_markdown_to_mindmap(md_path: str) -> Dict:
    """
    根据 markdown 文件生成思维导图树结构。

    返回的 dict 结构大致为：
    {
        "name": "... 根标题 ...",
        "source_file": md_path,
        "refs": [{"line": 1, "endLine": last_line}],
        "level": 0,
        "slug": "...",
        "children": [
            {
                "name": "Abstract",
                "refs": [{"line": 13, "endLine": 16}],
                "level": 1,
                "slug": "...",
                "children": [...],
                "content": "... 原文片段 ..."
            },
            ...
        ],
        "content": "... 整篇原文 ..."
    }
    """
    # ---------- 读文件 + 带行号 ----------
    lines: List[Tuple[int, str]] = []
    with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, l in enumerate(f, start=1):
            lines.append((i, l.rstrip("\n")))

    if not lines:
        title = os.path.basename(md_path)
        return {
            "name": title,
            "source_file": md_path,
            "refs": [],
            "children": [],
            "level": 0,
            "slug": _slugify(title),
            "content": "",
        }

    # ---------- 找根标题（第一个 # 开头的行） ----------
    title = None
    title_line = None
    for ln, txt in lines:
        if txt.startswith("#"):
            title = txt.lstrip("#").strip()
            title_line = ln
            break
    if not title:
        title = os.path.basename(md_path)

    last_line_no = lines[-1][0]

    root: Dict = {
        "name": title,
        "source_file": md_path,
        "refs": [{"line": 1, "endLine": last_line_no}],
        "children": [],
        "level": 0,
        "slug": _slugify(title),
    }

    # ---------- 收集所有标题（除了根标题） ----------
    headings: List[Dict] = []
    for ln, txt in lines:
        if not txt.startswith("#"):
            continue
        if ln == title_line:
            # 根标题已经作为 root.name，用来当中心节点，不再重复加子节点
            continue
        headings.append({"line": ln, "raw": txt})

    # 为每个 heading 计算 endLine 和 name
    for i, h in enumerate(headings):
        start = h["line"]
        if i + 1 < len(headings):
            end = headings[i + 1]["line"] - 1
        else:
            end = last_line_no
        h["endLine"] = end
        h["name"] = h["raw"].lstrip("#").strip()

    # ---------- 构建树结构 ----------
    numeric_nodes: Dict[str, Dict] = {}    # "3.1" -> node
    last_by_depth: Dict[int, Dict] = {}    # depth -> 最近的该深度章节节点
    children: List[Dict] = []              # root.children

    for h in headings:
        name = h["name"]
        node: Dict = {
            "name": name,
            "refs": [{"line": h["line"], "endLine": h["endLine"]}],
            "children": [],
            "slug": _slugify(name),
        }

        # 1) Abstract / 摘要：始终作为顶层第一个 child
        lower = name.lower()
        if lower.startswith("abstract") or name.startswith("摘要"):
            node["level"] = 1
            # 插在最前面（如果已经有的话就再往前插一位）
            children.insert(0, node)
            continue

        # 2) 带数字章节号的标题
        key = _extract_section_key(name)
        if key:
            depth = len(key.split("."))
            node["level"] = depth
            numeric_nodes[key] = node

            if depth == 1:
                # 一级章节直接挂根
                children.append(node)
            else:
                # 找父章节（如 3.1 -> 3）
                parent_key = ".".join(key.split(".")[:-1])
                parent = numeric_nodes.get(parent_key)

                # 如果刚好没找到父章节，就退到最近浅层章节
                if parent is None:
                    for d in range(depth - 1, 0, -1):
                        # 找一个深度为 d 的最近章节
                        candidate = None
                        for k, n in numeric_nodes.items():
                            if len(k.split(".")) == d:
                                candidate = n
                        if candidate is not None:
                            parent = candidate
                            break

                if parent is None:
                    children.append(node)
                else:
                    parent.setdefault("children", []).append(node)

            # 更新“最近章节”
            last_by_depth[depth] = node
            # 清掉更深层的“最近章节”记录
            for d in list(last_by_depth.keys()):
                if d > depth:
                    del last_by_depth[d]

            continue

        # 3) 无章节号的小标题：挂到最近的 depth>=2 的章节下
        candidate = None
        max_depth = 0
        for d, n in last_by_depth.items():
            if d >= 2 and d > max_depth:
                max_depth = d
                candidate = n

        if candidate is not None:
            node["level"] = candidate["level"] + 1
            candidate.setdefault("children", []).append(node)
        else:
            # 没有任何 depth>=2 的章节，就当顶层节点挂 root
            node["level"] = 1
            children.append(node)

    root["children"] = children

    # ---------- 填充 content（可选） ----------
    line_map = {ln: txt for ln, txt in lines}

    def _fill_content(n: Dict):
        r = n.get("refs", [])
        if r:
            start, end = r[0]["line"], r[0]["endLine"]
            n["content"] = "\n".join(line_map[i] for i in range(start, end + 1))
        else:
            n["content"] = ""
        for c in n.get("children", []):
            _fill_content(c)

    _fill_content(root)

    return root


def parse_content_list_to_mindmap(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    title = os.path.basename(os.path.dirname(json_path))
    root = {"name": title, "source_file": json_path, "refs": [], "children": [], "level": 0, "slug": _slugify(title)}
    if isinstance(data, list):
        for item in data[:50]:
            name = str(item.get('title') or item.get('name') or item.get('section') or item)[:50]
            node = {"name": name, "refs": [], "children": [], "level": 1, "slug": _slugify(name)}
            root["children"].append(node)
    elif isinstance(data, dict):
        for k in list(data.keys())[:50]:
            node = {"name": str(k)[:50], "refs": [], "children": [], "level": 1, "slug": _slugify(str(k))}
            root["children"].append(node)
    return root

def save_mindmap(dataset_name: str, mindmap: Dict) -> str:
    out_dir = os.path.join("data", "uploaded", dataset_name, "mineru_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mindmap.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mindmap, f, ensure_ascii=False, indent=2)
    return out_path

def _slugify(text: str) -> str:
    s = re.sub(r"[\s\t\n]+", "_", text.strip())
    s = re.sub(r"[^\w\-\u4e00-\u9fff]", "", s)
    return s[:60] or "section"

def _fill_node_contents(node: Dict, lines: List[tuple]):
    refs = node.get("refs") or []
    content = ""
    if refs:
        start = max(1, refs[0].get("line", 1))
        end = max(start, refs[0].get("endLine", start))
        buf = []
        for ln, txt in lines:
            if ln < start:
                continue
            if ln > end:
                break
            buf.append(txt)
        text = "\n".join([b for b in buf if b.strip()])
        if text:
            content = text[:1200]
    node["content"] = content
    for ch in node.get("children", []):
        _fill_node_contents(ch, lines)

def _extract_section_key(name: str) -> str:
    s = (name or "").strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)+)\b", s)
    if m:
        return m.group(1)
    m2 = re.match(r"^\s*([0-9]+)\b", s)
    if m2:
        return m2.group(1)
    return ""


def _sort_tree_by_section_numbers(node: Dict):
    def norm_key(name: str) -> str:
        s = (name or "").strip()
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*)\b", s)
        return m.group(1) if m else ""
    ch = node.get("children", [])
    def key_fn(x: Dict):
        k = norm_key(x.get("name", ""))
        if not k:
            return (float('inf'), x.get("name", ""))
        try:
            return tuple(int(t) for t in k.split('.'))
        except Exception:
            return (float('inf'), x.get("name", ""))
    try:
        ch.sort(key=key_fn)
    except Exception:
        pass
    for c in ch:
        _sort_tree_by_section_numbers(c)
    return node

def _promote_special_sections(root: Dict) -> Dict:
    specials = {
        "abstract": None,
        "acknowledgments": None,
        "references": None,
    }
    def capture(parent: Dict):
        children = parent.get("children", [])
        i = 0
        while i < len(children):
            n = children[i]
            name_lc = (n.get("name", "").strip().lower())
            key = None
            for k in specials.keys():
                if k in name_lc:
                    key = k
                    break
            if key:
                specials[key] = n
                children.pop(i)
                continue
            capture(n)
            i += 1
    capture(root)
    # After removal, insert specials at desired positions: Abstract at start, Ack & References at end
    def set_level(n: Dict, lvl: int):
        if not n:
            return
        n["level"] = lvl
        for ch in n.get("children", []):
            set_level(ch, lvl + 1)
    # Sort current children by numbers first
    _sort_tree_by_section_numbers(root)
    ch = list(root.get("children", []))
    new_ch = []
    if specials["abstract"]:
        set_level(specials["abstract"], 1)
        new_ch.append(specials["abstract"])
    new_ch.extend(ch)
    if specials["acknowledgments"]:
        set_level(specials["acknowledgments"], 1)
        new_ch.append(specials["acknowledgments"])
    if specials["references"]:
        set_level(specials["references"], 1)
        new_ch.append(specials["references"])
    root["children"] = new_ch
    return root

def materialize_top_level_sections(dataset_name: str, mindmap: Dict) -> List[str]:
    compiled = compile_mindmap_md(dataset_name, mindmap)
    return [os.path.basename(compiled)]

def _refill_contents_from_source(mindmap: Dict):
    src = mindmap.get("source_file")
    if not src or not os.path.exists(src):
        return
    lines = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f, start=1):
            lines.append((i, l.rstrip()))
    _fill_node_contents(mindmap, lines)

def compile_mindmap_md(dataset_name: str, mindmap: Dict) -> str:
    out_dir = os.path.join("data", "uploaded", dataset_name, "mineru_out")
    os.makedirs(out_dir, exist_ok=True)
    compiled = os.path.join(out_dir, "compiled_mindmap.md")
    lines = []
    title = mindmap.get("name") or dataset_name
    lines.append(f"# {title}\n")
    def heading(level: int, text: str) -> str:
        return ("#" * max(1, level)) + f" {text}\n\n"
    def walk(n: Dict, lvl: int = 2):
        name = n.get("name") or "Section"
        lines.append(heading(lvl, name))
        content = n.get("content") or ""
        if content:
            lines.append(content + "\n\n")
        cp = n.get("content_path")
        if cp:
            lines.append(f"参考：{cp}\n\n")
        for ch in n.get("children", []):
            walk(ch, min(lvl + 1, 6))
    for ch in mindmap.get("children", []):
        walk(ch, 2)
    with open(compiled, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    return compiled

def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format"""
    nodes_dict = {}
    links = []
    
    # Extract nodes and relationships from the list
    for item in graph_data:
        if not isinstance(item, dict):
            continue
            
        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")
        
        # Helper: fallback to a reasonable node id/name when properties.name is missing
        def _fallback_node_id(node: Dict) -> str:
            props = node.get("properties", {}) or {}
            name = props.get("name") or props.get("summary") or props.get("caption") or props.get("schema_type")
            if isinstance(name, (list, dict)):
                name = str(name)
            label = node.get("label", "entity")
            chunk_id = props.get("chunk id")
            # Compose a stable readable identifier
            candidate = name or (f"{label}_{chunk_id}" if chunk_id else label)
            return str(candidate) if candidate else label
        
        # Process start node
        start_id = ""
        end_id = ""
        if start_node:
            start_id = _fallback_node_id(start_node)
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": str(start_id)[:30],
                    "category": start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {})
                }
        
        # Process end node
        if end_node:
            end_id = _fallback_node_id(end_node)
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": str(end_id)[:30],
                    "category": end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {})
                }
        
        # Add relationship
        if start_id and end_id:
            links.append({
                "source": start_id,
                "target": end_id,
                "name": relation,
                "value": 1
            })
    
    # Create categories
    categories_set = set()
    for node in nodes_dict.values():
        categories_set.add(node["category"])
    
    categories = []
    for i, cat_name in enumerate(categories_set):
        categories.append({
            "name": cat_name,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(categories_set)}, 70%, 60%)"
            }
        })
    
    nodes = list(nodes_dict.values())
    
    return {
        "nodes": nodes[:500],  # Limit for better visual effects​​
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }

def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert standard {nodes: [], edges: []} format to ECharts format"""
    nodes = []
    links = []
    categories = []
    
    # Extract unique categories
    node_types = set()
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "entity")
        node_types.add(node_type)
    
    for i, node_type in enumerate(node_types):
        categories.append({
            "name": node_type,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"
            }
        })
    
    # Process nodes
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": node.get("type", "entity"),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", [])
        })
    
    # Process edges
    for edge in graph_data.get("edges", []):
        links.append({
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "name": edge.get("relation", "related_to"),
            "value": edge.get("weight", 1)
        })
    
    return {
        "nodes": nodes[:500],  # Limit for performance
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }

@app.post("/api/mindmap/generate", response_model=MindmapGenerateResponse)
async def generate_mindmap(request: MindmapGenerateRequest):
    try:
        _setup_mindmap_logger()
        dataset_name = request.dataset_name
        base_dir = os.path.join("data", "uploaded", dataset_name)
        if not os.path.exists(base_dir):
            raise HTTPException(status_code=404, detail="Dataset not found")
        # Idempotent: reuse existing mindmap.json if present
        out_mindmap = os.path.join(base_dir, "mineru_out", "mindmap.json")
        if os.path.exists(out_mindmap):
            with open(out_mindmap, 'r', encoding='utf-8') as f:
                mindmap = json.load(f)
            if request.materialize:
                try:
                    src_file = mindmap.get("source_file")
                    if src_file and os.path.isfile(src_file):
                        _refill_contents_from_source(mindmap)
                except Exception:
                    pass
                materialize_top_level_sections(dataset_name, mindmap)
                save_mindmap(dataset_name, mindmap)
            return MindmapGenerateResponse(success=True, message="Mindmap loaded", dataset_name=dataset_name, mindmap=mindmap, save_path=out_mindmap)

        src = None
        md_files = list_mineru_md_files(dataset_name)
        if not md_files:
            src = find_mineru_source_file(dataset_name)
            if not src:
                raise HTTPException(status_code=404, detail="No MinerU source found")
            if src.endswith('.md'):
                mindmap = parse_markdown_to_mindmap(src)
            else:
                mindmap = parse_content_list_to_mindmap(src)
        else:
            root_title = dataset_name
            root = {"name": root_title, "source_file": base_dir, "refs": [], "children": [], "level": 0, "slug": _slugify(root_title)}
            for p in md_files:
                try:
                    mm = parse_markdown_to_mindmap(p)
                    mm["level"] = 1
                    root.setdefault("children", []).append(mm)
                except Exception:
                    continue
            mindmap = root
        if request.materialize:
            try:
                src_file = mindmap.get("source_file")
                if src_file and os.path.isfile(src_file):
                    _refill_contents_from_source(mindmap)
            except Exception:
                pass
            materialize_top_level_sections(dataset_name, mindmap)
        save_path = save_mindmap(dataset_name, mindmap)
        try:
            logger.info(f"mindmap generated: dataset='{dataset_name}' src='{src or base_dir}' saved='{save_path}'")
        except Exception:
            pass
        return MindmapGenerateResponse(success=True, message="Mindmap generated", dataset_name=dataset_name, mindmap=mindmap, save_path=save_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MindmapMaterializeRequest(BaseModel):
    dataset_name: str

@app.post("/api/mindmap/materialize")
async def materialize_mindmap(request: MindmapMaterializeRequest):
    try:
        _setup_mindmap_logger()
        dataset_name = request.dataset_name
        out_path = os.path.join("data", "uploaded", dataset_name, "mineru_out", "mindmap.json")
        if not os.path.exists(out_path):
            raise HTTPException(status_code=404, detail="Mindmap not found")
        with open(out_path, 'r', encoding='utf-8') as f:
            mindmap = json.load(f)
        # Only refill when root source is a file
        try:
            src_file = mindmap.get("source_file")
            if src_file and os.path.isfile(src_file):
                _refill_contents_from_source(mindmap)
        except Exception:
            pass
        saved = materialize_top_level_sections(dataset_name, mindmap)
        save_mindmap(dataset_name, mindmap)
        return {"success": True, "saved": saved}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap materialize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mindmap/{dataset_name}")
async def get_mindmap(dataset_name: str):
    try:
        _setup_mindmap_logger()
        out_path = os.path.join("data", "uploaded", dataset_name, "mineru_out", "mindmap.json")
        if not os.path.exists(out_path):
            raise HTTPException(status_code=404, detail="Mindmap not found")
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap read failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _mindmap_to_echarts_tree(node: Dict) -> Dict:
    item = {
        "name": node.get("name", ""),
        "children": [],
        "value": len((node.get("content") or "").strip()),
        "extra": {
            "level": node.get("level", 0),
            "slug": node.get("slug"),
            "refs": node.get("refs", []),
            "content_path": node.get("content_path"),
        }
    }
    for ch in node.get("children", [])[:200]:
        item["children"].append(_mindmap_to_echarts_tree(ch))
    return item

def _build_section_hierarchy(root: Dict) -> Dict:
    base = {"name": root.get("name"), "children": [], "refs": root.get("refs", []), "level": 0, "slug": root.get("slug"), "content_path": root.get("content_path"), "content": root.get("content")}
    children = root.get("children", [])
    # Normalize keys: capture leading number-dot chain, without trailing dot
    def norm_key(name: str) -> str:
        s = (name or "").strip()
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*)\b", s)
        return m.group(1) if m else ""
    # Build node map for direct children
    key_node: Dict[str, Dict] = {}
    order: Dict[str, int] = {}
    non_numeric = []
    for idx, ch in enumerate(children):
        k = norm_key(ch.get("name", ""))
        if k:
            node = dict(ch)
            node.setdefault("children", [])
            key_node[k] = node
            order[k] = idx
        else:
            non_numeric.append(ch)
    # Attach numbered nodes under their longest existing parent
    for k in sorted(key_node.keys(), key=lambda x: (len(x.split('.')), order[x])):
        parts = k.split('.')
        if len(parts) == 1:
            base["children"].append(key_node[k])
            continue
        # find longest existing parent
        parent = None
        for d in range(len(parts)-1, 0, -1):
            pk = '.'.join(parts[:d])
            if pk in key_node:
                parent = key_node[pk]
                break
        if parent is None:
            base["children"].append(key_node[k])
        else:
            parent.setdefault("children", []).append(key_node[k])
    # Append non-numeric nodes
    base["children"].extend(non_numeric)
    # Recursively sort children by numeric section keys
    def norm_key(name: str) -> str:
        s = (name or "").strip()
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*)\b", s)
        return m.group(1) if m else ""
    def sort_children(n: Dict):
        ch = n.get("children", [])
        def key_fn(x):
            k = norm_key(x.get("name", ""))
            if not k:
                return (float('inf'), )
            try:
                return tuple(int(t) for t in k.split('.'))
            except Exception:
                return (float('inf'), )
        try:
            ch.sort(key=key_fn)
        except Exception:
            pass
        for c in ch:
            sort_children(c)
    sort_children(base)
    return base

def _load_results_map(dataset_name: str) -> Dict[str, Dict]:
    m = {}
    base = os.path.join("output", "mindmap_qa", dataset_name)
    if not os.path.exists(base):
        return m
    for f in os.listdir(base):
        if not f.endswith(".json"):
            continue
        p = os.path.join(base, f)
        try:
            with open(p, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            slug = os.path.splitext(f)[0]
            m[slug] = data
        except Exception:
            continue
    return m

def _load_md_results_map(dataset_name: str, src: str, filename: str) -> Dict[str, Dict]:
    m = {}
    base = os.path.join("output", "mindmap_qa_md", dataset_name, src.strip().lower(), os.path.splitext(os.path.basename(filename))[0])
    if not os.path.exists(base):
        return m
    for f in os.listdir(base):
        if not f.endswith(".json"):
            continue
        p = os.path.join(base, f)
        try:
            with open(p, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            slug = os.path.splitext(f)[0]
            m[slug] = data
        except Exception:
            continue
    return m

def _attach_results_to_tree(node: Dict, results: Dict[str, Dict]):
    slug = node.get("slug") or _slugify(node.get("name", ""))
    res = results.get(slug)
    if res:
        node["results"] = {
            "explanation": res.get("explanation"),
            "evidence_triples": res.get("evidence_triples", []),
            "evidence_chunks": res.get("evidence_chunks", []),
            "coverage": res.get("coverage", 0),
        }
    for ch in node.get("children", []):
        _attach_results_to_tree(ch, results)

def _mindmap_to_echarts_tree_with_results(node: Dict) -> Dict:
    item = {
        "name": node.get("name", ""),
        "children": [],
        "value": len(((node.get("results", {}) or {}).get("explanation") or (node.get("content") or "")).strip()),
        "extra": {
            "level": node.get("level", 0),
            "slug": node.get("slug"),
            "refs": node.get("refs", []),
            "content_path": node.get("content_path"),
            "results": node.get("results") or {}
        }
    }
    for ch in node.get("children", [])[:200]:
        item["children"].append(_mindmap_to_echarts_tree_with_results(ch))
    return item

@app.get("/api/mindmap/visualization/{dataset_name}")
async def get_mindmap_visualization(dataset_name: str):
    try:
        _setup_mindmap_logger()
        out_path = os.path.join("data", "uploaded", dataset_name, "mineru_out", "mindmap.json")
        if not os.path.exists(out_path):
            raise HTTPException(status_code=404, detail="Mindmap not found")
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Build a hierarchical view for visualization (does not mutate saved JSON)
        tree_struct = _build_section_hierarchy(data)
        tree = _mindmap_to_echarts_tree(tree_struct)
        def _count(n):
            c = 1
            for ch in n.get("children", []):
                c += _count(ch)
            return c
        stats = {"nodes": _count(tree), "depth": 0}
        def _depth(n, d):
            dd = d
            for ch in n.get("children", []):
                dd = max(dd, _depth(ch, d+1))
            return dd
        stats["depth"] = _depth(tree, 1)
        return {"data": [tree], "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mindmap-tree/{dataset_name}")
async def get_mindmap_tree(dataset_name: str):
    try:
        _setup_mindmap_logger()
        src = find_mineru_source_file(dataset_name)
        if not src or not os.path.isfile(src):
            raise HTTPException(status_code=404, detail="MinerU md not found. Please run MinerU to produce <dataset>.md under ocr/ or vlm/")
        data = parse_markdown_to_mindmap(src)
        data = _promote_special_sections(data)
        tree = _mindmap_to_echarts_tree(_sort_tree_by_section_numbers(data))
        return {"data": [tree]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap tree failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mindmap/qa/tree/{dataset_name}")
async def get_mindmap_results_tree(dataset_name: str):
    try:
        _setup_mindmap_logger()
        out_path = os.path.join("data", "uploaded", dataset_name, "mineru_out", "mindmap.json")
        if not os.path.exists(out_path):
            raise HTTPException(status_code=404, detail="Mindmap not found")
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tree_struct = _build_section_hierarchy(data)
        results = _load_results_map(dataset_name)
        _attach_results_to_tree(tree_struct, results)
        tree = _mindmap_to_echarts_tree_with_results(tree_struct)
        def _count(n):
            c = 1
            for ch in n.get("children", []):
                c += _count(ch)
            return c
        stats = {"nodes": _count(tree), "with_results": len(results)}
        return {"data": [tree], "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap results tree failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mindmap/qa/md/tree/{dataset_name}/{src}/{filename}")
async def get_mindmap_md_results_tree(dataset_name: str, src: str, filename: str):
    try:
        _setup_mindmap_logger()
        base_dir = os.path.join("data", "uploaded", dataset_name, "mineru_out")
        md_file = os.path.join(base_dir, src.strip().lower(), filename)
        if not os.path.exists(md_file):
            alt = find_mineru_source_file(dataset_name)
            if not alt or not alt.endswith('.md'):
                raise HTTPException(status_code=404, detail="MinerU md not found")
            md_file = alt
        data = parse_markdown_to_mindmap(md_file)
        data = _promote_special_sections(data)
        tree_struct = _sort_tree_by_section_numbers(data)
        results = _load_md_results_map(dataset_name, src, filename)
        _attach_results_to_tree(tree_struct, results)
        tree = _mindmap_to_echarts_tree_with_results(tree_struct)
        def _count(n):
            c = 1
            for ch in n.get("children", []):
                c += _count(ch)
            return c
        stats = {"nodes": _count(tree), "with_results": len(results)}
        return {"data": [tree], "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    """Process question using agent mode (iterative retrieval + reasoning) and return answer."""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name
        question = request.question

        await send_progress_update(client_id, "retrieval", 10, "Initializing retrieval system (agent mode)...")

        graph_path = f"output/graphs/{dataset_name}_new.json"
        schema_path = get_schema_path_for_dataset(dataset_name)
        if not os.path.exists(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")

        # Config & components
        global config
        if config is None:
            config = get_config("config/base_config.yaml")

        graphq = decomposer.GraphQ(dataset_name, config=config)
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",  # force agent mode
            config=config
        )

        await send_progress_update(client_id, "retrieval", 40, "Building indices...")
        loop = asyncio.get_running_loop()
        # Offload index building to thread executor to avoid blocking event loop
        await loop.run_in_executor(None, kt_retriever.build_indices)

        # Notify QA start via WS so frontend can show immediate progress
        try:
            await manager.send_message({
                "type": "qa_update",
                "stage": "start",
                "message": "Question processing started",
                "dataset": dataset_name,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }, client_id)
            await asyncio.sleep(0)
        except Exception as _e:
            logger.debug(f"QA start ws send failed: {_e}")

        # Helper functions (reuse a simplified version of main.py logic)
        def _dedup(items):
            return list({x: None for x in items}.keys())
        def _merge_chunk_contents(ids, mapping):
            return [mapping.get(i, f"[Missing content for chunk {i}]") for i in ids]

        # Step 1: decomposition
        await send_progress_update(client_id, "retrieval", 50, "Decomposing question...")
        try:
            # Offload decomposition to executor
            loop = asyncio.get_running_loop()
            decomposition = await loop.run_in_executor(None, lambda: graphq.decompose(question, schema_path))
            sub_questions = decomposition.get("sub_questions", [])
            involved_types = decomposition.get("involved_types", {})
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "decompose",
                    "sub_questions_count": len(sub_questions),
                    "sub_questions": [sq.get("sub-question", "") for sq in sub_questions][:5],
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                await asyncio.sleep(0.05)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Decompose failed: {e}")
            sub_questions = [{"sub-question": question}]
            involved_types = {"nodes": [], "relations": [], "attributes": []}
            decomposition = {"sub_questions": sub_questions, "involved_types": involved_types}

        reasoning_steps = []
        all_triples = set()
        all_chunk_ids = set()
        all_chunk_contents: Dict[str, str] = {}

        # Step 2: initial retrieval for each sub-question
        await send_progress_update(client_id, "retrieval", 65, "Initial retrieval...")
        import time as _time
        for idx, sq in enumerate(sub_questions):
            sq_text = sq.get("sub-question", question)
            start_t = _time.time()
            # Offload retrieval to thread executor to avoid blocking event loop
            def _run_retrieval():
                return kt_retriever.process_retrieval_results(
                    sq_text,
                    top_k=config.retrieval.top_k_filter,
                    involved_types=involved_types
                )
            retrieval_results, elapsed = await loop.run_in_executor(None, _run_retrieval)
            triples = retrieval_results.get('triples', []) or []
            chunk_ids = retrieval_results.get('chunk_ids', []) or []
            chunk_contents = retrieval_results.get('chunk_contents', []) or []
            if isinstance(chunk_contents, dict):
                for cid, ctext in chunk_contents.items():
                    all_chunk_contents[cid] = ctext
            else:
                for i_c, cid in enumerate(chunk_ids):
                    if i_c < len(chunk_contents):
                        all_chunk_contents[cid] = chunk_contents[i_c]
            all_triples.update(triples)
            all_chunk_ids.update(chunk_ids)
            reasoning_steps.append({
                "type": "sub_question",
                "question": sq_text,
                "triples": triples[:10],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "chunk_contents": list(all_chunk_contents.values())[:3]
            })

            # Stream this sub-question's partial result to frontend via WebSocket
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "sub_question",
                    "index": idx + 1,
                    "total": len(sub_questions),
                    "question": sq_text,
                    "triples_preview": list(dict.fromkeys(triples))[:5],
                    "triples_count": len(triples),
                    "chunks_count": len(chunk_ids),
                    "processing_time": elapsed,
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                # yield to event loop to flush WS frames
                await asyncio.sleep(0)
            except Exception as _e:
                logger.debug(f"QA update ws send failed for sub_question {idx+1}: {_e}")

        initial_triples = _dedup(list(all_triples))
        initial_chunk_ids = list(set(all_chunk_ids))
        initial_chunk_contents = _merge_chunk_contents(initial_chunk_ids, all_chunk_contents)
        context_initial = "=== Triples ===\n" + "\n".join(initial_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(initial_chunk_contents[:10])
        init_prompt = kt_retriever.generate_prompt(question, context_initial)
        try:
            initial_answer = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(init_prompt))
        except Exception as e:
            initial_answer = f"Initial answer failed: {e}"
        final_answer = initial_answer

        enable_ircot = bool(getattr(getattr(config, 'retrieval', object()), 'agent', None) and getattr(getattr(config.retrieval, 'agent', object()), 'enable_ircot', False))
        if enable_ircot:
            await send_progress_update(client_id, "retrieval", 75, "Iterative reasoning...")
            try:
                await manager.send_message({
                    "type": "qa_update",
                    "stage": "ircot_start",
                    "message": "Starting iterative reasoning",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                await asyncio.sleep(0.05)
            except Exception:
                pass
            max_steps = getattr(getattr(config.retrieval, 'agent', object()), 'max_steps', 3)
            current_query = question
            thoughts = []
            thoughts.append(f"Initial: {initial_answer[:200]}")
            for step in range(1, max_steps + 1):
                loop_triples = _dedup(list(all_triples))
                loop_chunk_ids = list(set(all_chunk_ids))
                loop_chunk_contents = _merge_chunk_contents(loop_chunk_ids, all_chunk_contents)
                loop_ctx = "=== Triples ===\n" + "\n".join(loop_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(loop_chunk_contents[:10])
                loop_prompt = f"You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.\nCurrent Question: {question}\nCurrent Iteration Query: {current_query}\nKnowledge Context:\n{loop_ctx}\nPrevious Thoughts: {' | '.join(thoughts) if thoughts else 'None'}\nInstructions:\n1. If enough info answer with: So the answer is: <answer>\n2. Else propose new query with: The new query is: <query>\nYour reasoning:"
                try:
                    reasoning = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(loop_prompt))
                except Exception as e:
                    reasoning = f"Reasoning error: {e}"
                reasoning_steps.append({
                    "type": "ircot_step",
                    "question": current_query,
                    "triples": loop_triples[:10],
                    "triples_count": len(loop_triples),
                    "chunks_count": len(loop_chunk_ids),
                    "processing_time": 0,
                    "chunk_contents": loop_chunk_contents[:3],
                    "thought": (reasoning or "")[:300]
                })
                try:
                    await manager.send_message({
                        "type": "qa_update",
                        "stage": "ircot",
                        "step": step,
                        "max_steps": max_steps,
                        "current_query": current_query,
                        "thought_preview": (reasoning or "")[:200],
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
                    await asyncio.sleep(0)
                except Exception:
                    pass
                if "So the answer is:" in reasoning:
                    m = re.search(r"So the answer is:\s*(.*)", reasoning, flags=re.IGNORECASE | re.DOTALL)
                    final_answer = m.group(1).strip() if m else reasoning
                    break
                if "The new query is:" not in reasoning:
                    final_answer = initial_answer or reasoning
                    break
                new_query = reasoning.split("The new query is:", 1)[1].strip().splitlines()[0]
                if not new_query or new_query == current_query:
                    final_answer = initial_answer or reasoning
                    break
                current_query = new_query
                await send_progress_update(client_id, "retrieval", min(90, 75 + step * 5), f"Iterative retrieval step {step}...")
                try:
                    def _run_more_retrieval():
                        return kt_retriever.process_retrieval_results(current_query, top_k=config.retrieval.top_k_filter)
                    new_ret, _ = await loop.run_in_executor(None, _run_more_retrieval)
                    new_triples = new_ret.get('triples', []) or []
                    new_chunk_ids = new_ret.get('chunk_ids', []) or []
                    new_chunk_contents = new_ret.get('chunk_contents', []) or []
                    if isinstance(new_chunk_contents, dict):
                        for cid, ctext in new_chunk_contents.items():
                            all_chunk_contents[cid] = ctext
                    else:
                        for i_c, cid in enumerate(new_chunk_ids):
                            if i_c < len(new_chunk_contents):
                                all_chunk_contents[cid] = new_chunk_contents[i_c]
                    all_triples.update(new_triples)
                    all_chunk_ids.update(new_chunk_ids)
                except Exception:
                    break

        # Final aggregation
        final_triples = _dedup(list(all_triples))[:20]
        final_chunk_ids = list(set(all_chunk_ids))
        final_chunk_contents = _merge_chunk_contents(final_chunk_ids, all_chunk_contents)[:10]

        await send_progress_update(client_id, "retrieval", 100, "Answer generation completed!")

        # Notify frontend that QA process is complete with a compact summary
        try:
            await manager.send_message({
                "type": "qa_complete",
                "answer_preview": (final_answer or "")[:300],
                "sub_questions_count": len(sub_questions),
                "triples_final_count": len(final_triples),
                "chunks_final_count": len(final_chunk_contents),
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.debug(f"QA complete ws send failed: {_e}")

        visualization_data = {
            "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
            "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
            "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
            "retrieval_details": {
                "total_triples": len(final_triples),
                "total_chunks": len(final_chunk_contents),
                "sub_questions_count": len(sub_questions),
                "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if s.get("type") == "sub_question"]
            }
        }

        return QuestionResponse(
            answer=final_answer,
            sub_questions=sub_questions,
            retrieved_triples=final_triples,
            retrieved_chunks=final_chunk_contents,
            reasoning_steps=reasoning_steps,
            visualization_data=visualization_data
        )
    except Exception as e:
        await send_progress_update(client_id, "retrieval", 0, f"Question answering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def prepare_subquery_visualization(sub_questions: List[Dict], reasoning_steps: List[Dict]) -> Dict:
    """Prepare subquery visualization"""
    nodes = [{"id": "original", "name": "Original Question", "category": "question", "symbolSize": 40}]
    links = []
    
    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append({
            "id": sub_id,
            "name": sub_q.get("sub-question", "")[:20] + "...",
            "category": "sub_question",
            "symbolSize": 30
        })
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})
    
    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}}
        ]
    }

def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    """Prepare retrieved knowledge visualization"""
    nodes = []
    links = []
    node_set = set()
    
    for triple in triples[:10]:
        try:
            if triple.startswith('[') and triple.endswith(']'):
                try:
                    parts = ast.literal_eval(triple)
                except Exception:
                    continue
                if len(parts) == 3:
                    source, relation, target = parts
                    
                    for entity in [source, target]:
                        if entity not in node_set:
                            node_set.add(entity)
                            nodes.append({
                                "id": str(entity),
                                "name": str(entity)[:20],
                                "category": "entity",
                                "symbolSize": 20
                            })
                    
                    links.append({
                        "source": str(source),
                        "target": str(target),
                        "name": str(relation)
                    })
        except:
            continue
    
    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}]
    }

def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    """Prepare reasoning flow visualization"""
    steps_data = []
    for i, step in enumerate(reasoning_steps):
        steps_data.append({
            "step": i + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:50],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0)
        })
    
    return {
        "steps": steps_data,
        "timeline": [step["processing_time"] for step in steps_data]
    }

@app.get("/api/datasets")
async def get_datasets():
    """Get list of available datasets"""
    datasets = []
    
    # Check uploaded datasets
    upload_dir = "data/uploaded"
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isdir(item_path):
                corpus_path = os.path.join(item_path, "corpus.json")
                if os.path.exists(corpus_path):
                    graph_path = f"output/graphs/{item}_new.json"
                    status = "ready" if os.path.exists(graph_path) else "needs_construction"
                    has_custom_schema = os.path.exists(f"schemas/{item}.json")
                    datasets.append({
                        "name": item,
                        "type": "uploaded",
                        "status": status,
                        "has_custom_schema": has_custom_schema
                    })
    
    # Add demo dataset
    demo_corpus = "data/demo/demo_corpus.json"
    if os.path.exists(demo_corpus):
        demo_graph = "output/graphs/demo_new.json"
        status = "ready" if os.path.exists(demo_graph) else "needs_construction"
        datasets.append({
            "name": "demo",
            "type": "demo", 
            "status": status,
            "has_custom_schema": False
        })
    
    return {"datasets": datasets}

@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(dataset_name: str, schema_file: UploadFile = File(...)):
    """Upload a custom schema JSON for a dataset."""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot upload schema for demo dataset")
        if not schema_file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Schema file must be a .json file")

        content = await schema_file.read()
        try:
            data = json.loads(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Schema JSON must be an object")

        os.makedirs("schemas", exist_ok=True)
        save_path = f"schemas/{dataset_name}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"success": True, "message": "Schema uploaded successfully", "dataset_name": dataset_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload schema: {str(e)}")

@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its associated files"""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot delete demo dataset")
        
        deleted_files = []
        
        # Delete dataset directory
        dataset_dir = f"data/uploaded/{dataset_name}"
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            deleted_files.append(dataset_dir)
        
        # Delete graph file
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            deleted_files.append(graph_path)
        
        # Delete schema file (if dataset-specific)
        schema_path = f"schemas/{dataset_name}.json"
        if os.path.exists(schema_path):
            os.remove(schema_path)
            deleted_files.append(schema_path)
        
        # Delete cache files
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            deleted_files.append(cache_dir)
        
        # Delete chunk files
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            deleted_files.append(chunk_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
        
    return {
        "success": True,
        "message": f"Dataset '{dataset_name}' deleted successfully",
        "deleted_files": deleted_files
    }

def _mindmap_classify(name: str) -> str:
    s = (name or "").strip().lower()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*)\b", s)
    if m:
        key = m.group(1)
        if "." in key:
            return "detail"
        return "overview"
    if any(k in s for k in ["术语", "定义", "概念", "关键词"]):
        return "definition"
    if any(k in s for k in ["步骤", "流程", "方法", "框架"]):
        return "process"
    return "detail"

def _collect_nodes_in_order(tree: Dict) -> List[Dict]:
    out = []
    def walk(n, path):
        out.append({"node": n, "path": path})
        for ch in n.get("children", []):
            walk(ch, path + [ch.get("name", "")])
    walk(tree, [tree.get("name", "")])
    return out

def parse_mineru_md_to_modules(md_path: str) -> List[Dict]:
    tree = parse_markdown_to_mindmap(md_path)
    tree = _promote_special_sections(tree)
    tree = _sort_tree_by_section_numbers(tree)
    items = _collect_nodes_in_order(tree)
    modules = []
    for it in items:
        n = it["node"]
        lvl = int(n.get("level", 0))
        if lvl <= 0:
            continue
        modules.append(n)
    return modules

def _ensure_mindmap_qa_logger():
    try:
        logs_dir = "output/logs"
        os.makedirs(logs_dir, exist_ok=True)
        fh_paths = [getattr(h, 'baseFilename', None) for h in logger.handlers]
        if not any(p and p.endswith("mindmap_qa.log") for p in fh_paths):
            setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "mindmap_qa.log"))
    except Exception:
        pass

@app.post("/api/mindmap/qa", response_model=MindmapQAResponse)
async def mindmap_qa(request: MindmapQARequest, client_id: str = "web_client"):
    try:
        _ensure_mindmap_qa_logger()
        dataset_name = request.dataset_name
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available (missing faiss). Please install faiss-cpu and construct the graph.")
        out_mindmap = os.path.join("data", "uploaded", dataset_name, "mineru_out", "mindmap.json")
        if os.path.exists(out_mindmap):
            with open(out_mindmap, 'r', encoding='utf-8') as f:
                mindmap = json.load(f)
        else:
            src = find_mineru_source_file(dataset_name)
            if not src:
                raise HTTPException(status_code=404, detail="Mindmap not found")
            if src.endswith('.md'):
                mindmap = parse_markdown_to_mindmap(src)
            else:
                mindmap = parse_content_list_to_mindmap(src)
        tree_struct = _build_section_hierarchy(mindmap)
        nodes_all = _collect_nodes_in_order(tree_struct)
        nodes = []
        for item in nodes_all:
            n = item["node"]
            lvl = int(n.get("level", 0))
            parent = item["path"][0] if item["path"] else ""
            if lvl == 1:
                nodes.append(item)
        graph_path = f"output/graphs/{dataset_name}_new.json"
        schema_path = get_schema_path_for_dataset(dataset_name)
        if not os.path.exists(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")
        global config
        if config is None:
            if 'get_config' in globals():
                config = get_config("config/base_config.yaml")
            else:
                raise HTTPException(status_code=503, detail="Configuration loader unavailable. Ensure GraphRAG is installed and faiss is available.")
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",
            config=config
        )
        loop = asyncio.get_running_loop()
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "start", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        await loop.run_in_executor(None, kt_retriever.build_indices)
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "indices_built", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        save_dir = os.path.join("output", "mindmap_qa", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        saved = []
        total_triples = 0
        total_chunks = 0
        md_lines = []
        for item in nodes:
            n = item["node"]
            name = n.get("name", "")
            content = n.get("content", "")
            mtype = _mindmap_classify(name)
            q = (name or "").strip()
            if content:
                q = (q + " " + content[:200]).strip()
            try:
                logger.info(f"mindmap qa retrieval start dataset='{dataset_name}' module='{name}' qlen={len(q)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "retrieval_start", "dataset": dataset_name, "module": name, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            def _run_retrieval():
                return kt_retriever.process_retrieval_results(q, top_k=config.retrieval.top_k_filter)
            ret, _elapsed = await loop.run_in_executor(None, _run_retrieval)
            triples = ret.get('triples', []) or []
            chunk_ids = ret.get('chunk_ids', []) or []
            chunk_contents = ret.get('chunk_contents', []) or []
            if isinstance(chunk_contents, dict):
                contents = list(chunk_contents.values())
            else:
                contents = []
                for i_c, cid in enumerate(chunk_ids):
                    if i_c < len(chunk_contents):
                        contents.append(chunk_contents[i_c])
            try:
                logger.info(f"mindmap qa retrieval done dataset='{dataset_name}' module='{name}' triples={len(triples)} chunks={len(contents)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "retrieval_done", "dataset": dataset_name, "module": name, "triples_count": len(triples), "chunks_count": len(contents), "triples_preview": triples[:5], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            total_triples += len(triples)
            total_chunks += len(contents)
            ctx = "=== Triples ===\n" + "\n".join(triples[:20]) + "\n=== Chunks ===\n" + "\n".join(contents[:10])
            prompt_type = {
                "overview": "overview",
                "detail": "detail",
                "definition": "definition",
                "process": "process"
            }.get(mtype, "detail")
            try:
                template = config.get_prompt_formatted("mindmap", prompt_type,
                                                      module_title=name,
                                                      module_content=content or "",
                                                      dataset_name=dataset_name,
                                                      evidence_triples="\n".join(triples[:20]),
                                                      evidence_chunks="\n".join(contents[:10]))
            except Exception:
                template = kt_retriever.generate_prompt(q, ctx)
            try:
                logger.info(f"mindmap qa prompt built dataset='{dataset_name}' module='{name}' type='{prompt_type}' plen={len(template)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "prompt_built", "dataset": dataset_name, "module": name, "prompt_type": prompt_type, "prompt_preview": (template or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            try:
                text = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(template))
            except Exception as e:
                text = f"Failed to generate explanation: {e}"
            try:
                logger.info(f"mindmap qa llm done dataset='{dataset_name}' module='{name}' tlen={len(text)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "llm_done", "dataset": dataset_name, "module": name, "answer_preview": (text or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            segs = re.split(r"[。.;；.!?\n]", text or "")
            segs = [s.strip() for s in segs if s.strip()]
            cites = len(re.findall(r"\[(?:chunk|triple)\s*#?\w*", text or ""))
            cov = (cites / max(1, len(segs)))
            used = []
            for cid in chunk_ids[:10]:
                used.append(str(cid))
            item_json = {
                "module_title": name,
                "module_type": mtype,
                "explanation": text,
                "evidence_triples": triples[:20],
                "evidence_chunks": contents[:10],
                "citations": used,
                "coverage": cov,
                "prompt_type": prompt_type,
                "prompt": (template or "")[:4000]
            }
            slug = n.get("slug") or _slugify(name)
            fpath = os.path.join(save_dir, slug + ".json")
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(item_json, f, ensure_ascii=False, indent=2)
            saved.append(fpath)
            md_lines.append(f"# {name}\n\n{text}\n")
            try:
                await manager.send_message({"type": "mindmap_qa_update", "stage": "module_complete", "dataset": dataset_name, "module": name, "saved": fpath, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
        full_md = os.path.join(save_dir, "full.md")
        with open(full_md, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(md_lines))
        saved.append(full_md)
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "complete", "dataset": dataset_name, "modules": len(nodes), "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        return MindmapQAResponse(success=True, saved_files=saved, stats={"modules": len(nodes), "total_triples": total_triples, "total_chunks": total_chunks})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"mindmap qa failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@app.post("/api/mindmap/qa/md", response_model=MindmapQAMdResponse)
async def mindmap_qa_md(request: MindmapQAMdRequest, client_id: str = "web_client"):
    try:
        dataset_name = request.dataset_name
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available (missing faiss). Please install faiss-cpu and construct the graph.")
        base_dir = os.path.join("data", "uploaded", dataset_name, "mineru_out")
        subdir = request.src.strip().lower()
        md_file = os.path.join(base_dir, subdir, request.filename)
        if not os.path.exists(md_file):
            alt = find_mineru_source_file(dataset_name)
            if not alt or not alt.endswith(".md"):
                raise HTTPException(status_code=404, detail="MinerU md not found")
            md_file = alt
        modules = parse_mineru_md_to_modules(md_file)
        graph_path = f"output/graphs/{dataset_name}_new.json"
        schema_path = get_schema_path_for_dataset(dataset_name)
        if not os.path.exists(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")
        global config
        if config is None:
            if 'get_config' in globals():
                config = get_config("config/base_config.yaml")
            else:
                raise HTTPException(status_code=503, detail="Configuration loader unavailable. Ensure GraphRAG is installed and faiss is available.")
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",
            config=config
        )
        loop = asyncio.get_running_loop()
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "start", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        await loop.run_in_executor(None, kt_retriever.build_indices)
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "indices_built", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        out_dir = os.path.join("output", "mindmap_qa_md", dataset_name, subdir, os.path.splitext(os.path.basename(md_file))[0])
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        total_triples = 0
        total_chunks = 0
        md_lines = []
        for n in modules:
            name = n.get("name", "")
            content = n.get("content", "")
            mtype = _mindmap_classify(name)
            q = (name or "").strip()
            if content:
                q = (q + " " + content[:200]).strip()
            try:
                logger.info(f"mindmap qa md retrieval start dataset='{dataset_name}' module='{name}' qlen={len(q)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "retrieval_start", "dataset": dataset_name, "module": name, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            def _run_retrieval():
                return kt_retriever.process_retrieval_results(q, top_k=config.retrieval.top_k_filter)
            ret, _elapsed = await loop.run_in_executor(None, _run_retrieval)
            triples = ret.get('triples', []) or []
            chunk_ids = ret.get('chunk_ids', []) or []
            chunk_contents = ret.get('chunk_contents', []) or []
            if isinstance(chunk_contents, dict):
                contents = list(chunk_contents.values())
            else:
                contents = []
                for i_c, cid in enumerate(chunk_ids):
                    if i_c < len(chunk_contents):
                        contents.append(chunk_contents[i_c])
            t_k = int(request.top_k_triples or 12)
            c_k = int(request.top_k_chunks or 6)
            total_triples += len(triples)
            total_chunks += len(contents)
            try:
                logger.info(f"mindmap qa md retrieval done dataset='{dataset_name}' module='{name}' triples={len(triples)} chunks={len(contents)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "retrieval_done", "dataset": dataset_name, "module": name, "triples_count": len(triples), "chunks_count": len(contents), "triples_preview": triples[:5], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            ctx = "=== Triples ===\n" + "\n".join(triples[:t_k]) + "\n=== Chunks ===\n" + "\n".join(contents[:c_k])
            ptype = request.prompt_strategy or "auto"
            if ptype == "auto":
                prompt_type = {
                    "overview": "overview",
                    "detail": "detail",
                    "definition": "definition",
                    "process": "process"
                }.get(mtype, "detail")
            else:
                prompt_type = ptype
            try:
                template = config.get_prompt_formatted("mindmap", prompt_type,
                                                      module_title=name,
                                                      module_content=content or "",
                                                      dataset_name=dataset_name,
                                                      evidence_triples="\n".join(triples[:t_k]),
                                                      evidence_chunks="\n".join(contents[:c_k]))
            except Exception:
                template = kt_retriever.generate_prompt(q, ctx)
            try:
                logger.info(f"mindmap qa md prompt built dataset='{dataset_name}' module='{name}' type='{prompt_type}' plen={len(template)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "prompt_built", "dataset": dataset_name, "module": name, "prompt_type": prompt_type, "prompt_preview": (template or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            try:
                text = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(template))
            except Exception as e:
                text = f"Failed to generate explanation: {e}"
            try:
                logger.info(f"mindmap qa md llm done dataset='{dataset_name}' module='{name}' tlen={len(text)}")
                await manager.send_message({"type": "mindmap_qa_update", "stage": "llm_done", "dataset": dataset_name, "module": name, "answer_preview": (text or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            segs = re.split(r"[。.;；.!?\n]", text or "")
            segs = [s.strip() for s in segs if s.strip()]
            cites = len(re.findall(r"\[(?:chunk|triple)\s*#?\w*", text or ""))
            cov = (cites / max(1, len(segs)))
            used = []
            for cid in chunk_ids[:c_k]:
                used.append(str(cid))
            item_json = {
                "module_title": name,
                "module_type": mtype,
                "explanation": text,
                "evidence_triples": triples[:t_k],
                "evidence_chunks": contents[:c_k],
                "citations": used,
                "coverage": cov,
                "prompt_type": prompt_type,
                "prompt": (template or "")[:4000]
            }
            slug = n.get("slug") or _slugify(name)
            fpath = os.path.join(out_dir, slug + ".json")
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(item_json, f, ensure_ascii=False, indent=2)
            saved.append(fpath)
            md_lines.append(f"# {name}\n\n{text}\n")
            try:
                await manager.send_message({"type": "mindmap_qa_update", "stage": "module_complete", "dataset": dataset_name, "module": name, "saved": fpath, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
        full_md = os.path.join(out_dir, "full.md")
        with open(full_md, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(md_lines))
        saved.append(full_md)
        try:
            await manager.send_message({"type": "mindmap_qa_update", "stage": "complete", "dataset": dataset_name, "modules": len(modules), "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        return MindmapQAMdResponse(success=True, saved_files=saved, stats={"modules": len(modules), "total_triples": total_triples, "total_chunks": total_chunks})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    """Reconstruct graph for an existing dataset"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        # Check if dataset exists
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = "data/demo/demo_corpus.json"
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        await send_progress_update(client_id, "reconstruction", 5, "Starting reconstruction...")
        
        # Delete existing graph file
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "Old graph file deleted...")
        
        # Delete existing cache files
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "Cache files cleared...")
        
        await send_progress_update(client_id, "reconstruction", 35, "Reinitializing graph builder...")
        
        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")
        
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)
        
        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )
        
        await send_progress_update(client_id, "reconstruction", 50, "Rebuilding knowledge graph...")
        
        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Run graph reconstruction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
        
        await send_progress_update(client_id, "reconstruction", 100, "Graph reconstruction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "reconstruction",
                "message": "Graph reconstruction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")
        
        return {
            "success": True,
            "message": "Dataset reconstructed successfully",
            "dataset_name": dataset_name
        }
    
    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"Reconstruction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "reconstruction",
                "message": f"Reconstruction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    """Get graph visualization data"""
    graph_path = f"output/graphs/{dataset_name}_new.json"
    
    if not os.path.exists(graph_path):
        # Return demo data
        return {
            "nodes": [
                {"id": "node1", "name": "Example Entity 1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1}
        }
    
    return await prepare_graph_visualization(graph_path)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("data/uploaded", exist_ok=True)
    os.makedirs("output/graphs", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("schemas", exist_ok=True)
    
    logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
