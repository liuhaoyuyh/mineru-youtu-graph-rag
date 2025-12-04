import os
import json
import glob
import shutil
import asyncio
from typing import Dict

from fastapi import HTTPException

from utils.logger import logger, setup_logger
import logging

from app.core.ws import send_progress_update, get_manager
from app.utils.visualization import convert_graphrag_format, convert_standard_format


def ensure_demo_schema_exists() -> str:
    os.makedirs("schemas", exist_ok=True)
    schema_path = "schemas/demo.json"
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person",
                "location",
                "organization",
                "event",
                "object",
                "concept",
                "time_period",
                "creative_work",
                "biological_entity",
                "natural_phenomenon",
            ],
            "Relations": [
                "is_a",
                "part_of",
                "located_in",
                "created_by",
                "used_by",
                "participates_in",
                "related_to",
                "belongs_to",
                "influences",
                "precedes",
                "arrives_in",
                "comparable_to",
            ],
            "Attributes": [
                "name",
                "date",
                "size",
                "type",
                "description",
                "status",
                "quantity",
                "value",
                "position",
                "duration",
                "time",
            ],
        }
        with open(schema_path, "w") as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path


def get_schema_path_for_dataset(dataset_name: str) -> str:
    from app.services.dataset_service import get_schema_path_for_dataset as _get
    return _get(dataset_name)


async def clear_cache_files(dataset_name: str):
    try:
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")

        graph_file = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")

        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*",
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


def _is_graph_empty(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return True
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data) == 0
        if isinstance(data, dict):
            nodes = data.get("nodes") or []
            edges = data.get("edges") or data.get("links") or []
            return (len(nodes) == 0) and (len(edges) == 0)
        return True
    except Exception:
        return True


async def prepare_graph_visualization(graph_path: str) -> Dict:
    try:
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

        if isinstance(graph_data, list):
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}


    # functions moved to app.utils.visualization


async def construct_graph(dataset_name: str, client_id: str = "default") -> Dict:
    try:
        from app.core.graphrag import GRAPHRAG_AVAILABLE, get_config, constructor
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")

        logs_dir = "output/logs"
        os.makedirs(logs_dir, exist_ok=True)
        try:
            fh_paths = [getattr(h, "baseFilename", None) for h in logger.handlers]
            if not any(p and p.endswith("construction.log") for p in fh_paths):
                setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "construction.log"))
        except Exception:
            pass

        await send_progress_update(client_id, "construction", 2, "Cleaning old cache files...")
        await clear_cache_files(dataset_name)
        await send_progress_update(client_id, "construction", 5, "Initializing graph builder...")

        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
        schema_path = get_schema_path_for_dataset(dataset_name)
        if not os.path.exists(corpus_path):
            corpus_path = "data/demo/demo_corpus.json"
        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        await send_progress_update(client_id, "construction", 10, "Loading configuration and corpus...")
        config = get_config("config/base_config.yaml")

        try:
            from utils.multimodal_ingestion import build_chunks_for_dataset
            logger.info(f"multimodal ingestion start: dataset='{dataset_name}'")
            extra_chunks = build_chunks_for_dataset(dataset_name)
            logger.info(f"multimodal ingestion chunks={len(extra_chunks) if extra_chunks else 0}")
            if extra_chunks:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    base_docs = json.load(f)
                merged_docs = (base_docs or []) + extra_chunks
                os.makedirs(f"data/uploaded/{dataset_name}", exist_ok=True)
                merged_path = f"data/uploaded/{dataset_name}/merged_corpus.json"
                with open(merged_path, "w", encoding="utf-8") as f:
                    json.dump(merged_docs, f, ensure_ascii=False, indent=2)
                logger.info(f"merged_corpus written: path='{merged_path}', base_docs={len(base_docs) if base_docs else 0}, extra_chunks={len(extra_chunks)}")
                corpus_path = merged_path
        except Exception as _e:
            logger.warning(f"Multimodal ingestion skipped: {_e}")

        builder = constructor.KTBuilder(dataset_name, schema_path, mode=config.construction.mode, config=config)
        await send_progress_update(client_id, "construction", 20, "Starting entity-relation extraction...")

        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, build_graph_sync)

        await send_progress_update(client_id, "construction", 95, "Preparing visualization data...")
        graph_path = f"output/graphs/{dataset_name}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)

        await send_progress_update(client_id, "construction", 100, "Graph construction completed!")
        try:
            await get_manager().send_message({"type": "complete", "stage": "construction", "message": "Graph construction completed!", "timestamp": __import__("datetime").datetime.now().isoformat()}, client_id)
        except Exception:
            pass

        return graph_vis_data
    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"Construction failed: {str(e)}")
        try:
            await get_manager().send_message({"type": "error", "stage": "construction", "message": f"Construction failed: {str(e)}", "timestamp": __import__("datetime").datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        raise


async def reconstruct_dataset(dataset_name: str, client_id: str = "default") -> Dict:
    try:
        from app.core.graphrag import GRAPHRAG_AVAILABLE, get_config, constructor
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")

        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = "data/demo/demo_corpus.json"
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")
        try:
            cfg = get_config("config/base_config.yaml")
            merged_path = f"data/uploaded/{dataset_name}/merged_corpus.json"
            if os.path.exists(merged_path):
                corpus_path = merged_path
                await send_progress_update(client_id, "reconstruction", 10, "Using merged corpus for reconstruction...")
        except Exception:
            pass

        await send_progress_update(client_id, "reconstruction", 5, "Starting reconstruction...")
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "Old graph file deleted...")

        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "Cache files cleared...")

        await send_progress_update(client_id, "reconstruction", 35, "Reinitializing graph builder...")
        config = get_config("config/base_config.yaml")
        schema_path = get_schema_path_for_dataset(dataset_name)
        builder = constructor.KTBuilder(dataset_name, schema_path, mode=config.construction.mode, config=config)
        await send_progress_update(client_id, "reconstruction", 50, "Rebuilding knowledge graph...")

        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, build_graph_sync)
        await send_progress_update(client_id, "reconstruction", 100, "Graph reconstruction completed!")

        try:
            await get_manager().send_message({"type": "complete", "stage": "reconstruction", "message": "Graph reconstruction completed!", "timestamp": __import__("datetime").datetime.now().isoformat()}, client_id)
        except Exception:
            pass

        return {"success": True, "message": "Dataset reconstructed successfully", "dataset_name": dataset_name}
    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"Reconstruction failed: {str(e)}")
        try:
            await get_manager().send_message({"type": "error", "stage": "reconstruction", "message": f"Reconstruction failed: {str(e)}", "timestamp": __import__("datetime").datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        raise
