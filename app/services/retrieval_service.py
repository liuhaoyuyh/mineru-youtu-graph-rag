import os
import re
import ast
import asyncio
from datetime import datetime
from typing import Dict, List

from utils.logger import logger
from fastapi import HTTPException

from app.core.ws import send_progress_update, get_manager
from app.utils.visualization import (
    prepare_subquery_visualization,
    prepare_retrieved_graph_visualization,
    prepare_reasoning_flow_visualization,
)


async def ask_question(dataset_name: str, question: str, client_id: str = "default") -> Dict:
    from app.core.graphrag import GRAPHRAG_AVAILABLE, get_config, decomposer, retriever

    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")

    await send_progress_update(client_id, "retrieval", 10, "Initializing retrieval system (agent mode)...")
    graph_path = f"output/graphs/{dataset_name}_new.json"
    schema_path = _get_schema_path_for_dataset(dataset_name)
    if not os.path.exists(graph_path) or _is_graph_empty(graph_path):
        graph_path = "output/graphs/demo_new.json"
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")

    config = get_config("config/base_config.yaml")
    graphq = decomposer.GraphQ(dataset_name, config=config)
    kt_retriever = retriever.KTRetriever(
        dataset_name,
        graph_path,
        recall_paths=config.retrieval.recall_paths,
        schema_path=schema_path,
        top_k=config.retrieval.top_k_filter,
        mode="agent",
        config=config,
    )

    await send_progress_update(client_id, "retrieval", 40, "Building indices...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kt_retriever.build_indices)

    try:
        await get_manager().send_message({
            "type": "qa_update",
            "stage": "start",
            "message": "Question processing started",
            "dataset": dataset_name,
            "question": question,
            "timestamp": datetime.now().isoformat(),
        }, client_id)
    except Exception:
        pass

    def _dedup(items):
        return list({x: None for x in items}.keys())

    def _merge_chunk_contents(ids, mapping):
        return [mapping.get(i, f"[Missing content for chunk {i}]") for i in ids]

    await send_progress_update(client_id, "retrieval", 50, "Decomposing question...")
    try:
        decomposition = await loop.run_in_executor(None, lambda: graphq.decompose(question, schema_path))
        sub_questions = decomposition.get("sub_questions", [])
        involved_types = decomposition.get("involved_types", {})
        def _normalize_subquestions(sqs):
            norm = []
            for sq in sqs or []:
                if isinstance(sq, dict):
                    txt = sq.get("sub-question") or sq.get("sub_question") or sq.get("question") or sq.get("text") or ""
                    norm.append({"sub-question": txt})
                elif isinstance(sq, str):
                    norm.append({"sub-question": sq})
                else:
                    norm.append({"sub-question": str(sq)})
            return norm
        sub_questions = _normalize_subquestions(sub_questions)
        try:
            await get_manager().send_message({
                "type": "qa_update",
                "stage": "decompose",
                "sub_questions_count": len(sub_questions),
                "sub_questions": [sq.get("sub-question", "") for sq in sub_questions][:5],
                "timestamp": datetime.now().isoformat(),
            }, client_id)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Decompose failed: {e}")
        sub_questions = [{"sub-question": question}]
        involved_types = {"nodes": [], "relations": [], "attributes": []}

    reasoning_steps = []
    all_triples = set()
    all_chunk_ids = set()
    all_chunk_contents: Dict[str, str] = {}

    await send_progress_update(client_id, "retrieval", 65, "Initial retrieval...")
    import time as _time
    for idx, sq in enumerate(sub_questions):
        sq_text = sq.get("sub-question", question) if isinstance(sq, dict) else (sq if isinstance(sq, str) else question)
        start_t = _time.time()

        def _run_retrieval():
            return kt_retriever.process_retrieval_results(
                sq_text,
                top_k=config.retrieval.top_k_filter,
                involved_types=involved_types,
            )

        retrieval_results, elapsed = await loop.run_in_executor(None, _run_retrieval)
        triples = retrieval_results.get("triples", []) or []
        chunk_ids = retrieval_results.get("chunk_ids", []) or []
        chunk_contents = retrieval_results.get("chunk_contents", []) or []

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
            "chunk_contents": list(all_chunk_contents.values())[:3],
        })
        try:
            await get_manager().send_message({
                "type": "qa_update",
                "stage": "sub_question",
                "index": idx + 1,
                "total": len(sub_questions),
                "question": sq_text,
                "triples_preview": list(dict.fromkeys(triples))[:5],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "timestamp": datetime.now().isoformat(),
            }, client_id)
        except Exception:
            pass

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

    enable_ircot = bool(getattr(getattr(config, "retrieval", object()), "agent", None) and getattr(getattr(config.retrieval, "agent", object()), "enable_ircot", False))
    if enable_ircot:
        await send_progress_update(client_id, "retrieval", 75, "Iterative reasoning...")
        try:
            await get_manager().send_message({
                "type": "qa_update",
                "stage": "ircot_start",
                "message": "Starting iterative reasoning",
                "timestamp": datetime.now().isoformat(),
            }, client_id)
        except Exception:
            pass
        max_steps = getattr(getattr(config.retrieval, "agent", object()), "max_steps", 3)
        current_query = question
        thoughts = [f"Initial: {initial_answer[:200]}"]
        for step in range(1, max_steps + 1):
            loop_triples = _dedup(list(all_triples))
            loop_chunk_ids = list(set(all_chunk_ids))
            loop_chunk_contents = _merge_chunk_contents(loop_chunk_ids, all_chunk_contents)
            loop_ctx = "=== Triples ===\n" + "\n".join(loop_triples[:20]) + "\n=== Chunks ===\n" + "\n".join(loop_chunk_contents[:10])
            loop_prompt = (
                f"You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.\n"
                f"Current Question: {question}\n"
                f"Current Iteration Query: {current_query}\n"
                f"Knowledge Context:\n{loop_ctx}\n"
                f"Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}\n"
                "Instructions:\n"
                "1. If enough info answer with: So the answer is: <answer>\n"
                "2. Else propose new query with: The new query is: <query>\n"
                "Your reasoning:"
            )
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
                "thought": (reasoning or "")[:300],
            })
            try:
                await get_manager().send_message({
                    "type": "qa_update",
                    "stage": "ircot",
                    "step": step,
                    "max_steps": max_steps,
                    "current_query": current_query,
                    "thought_preview": (reasoning or "")[:200],
                    "timestamp": datetime.now().isoformat(),
                }, client_id)
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
                new_triples = new_ret.get("triples", []) or []
                new_chunk_ids = new_ret.get("chunk_ids", []) or []
                new_chunk_contents = new_ret.get("chunk_contents", []) or []
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

    final_triples = _dedup(list(all_triples))[:20]
    final_chunk_ids = list(set(all_chunk_ids))
    final_chunk_contents = _merge_chunk_contents(final_chunk_ids, all_chunk_contents)[:10]

    await send_progress_update(client_id, "retrieval", 100, "Answer generation completed!")
    try:
        await get_manager().send_message({
            "type": "qa_complete",
            "answer_preview": (final_answer or "")[:300],
            "sub_questions_count": len(sub_questions),
            "triples_final_count": len(final_triples),
            "chunks_final_count": len(final_chunk_contents),
            "tokens_total": int(getattr(kt_retriever, "token_len", 0)),
            "timestamp": datetime.now().isoformat(),
        }, client_id)
    except Exception:
        pass

    visualization_data = {
        "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
        "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
        "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
        "retrieval_details": {
            "total_triples": len(final_triples),
            "total_chunks": len(final_chunk_contents),
            "sub_questions_count": len(sub_questions),
            "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if s.get("type") == "sub_question"],
        },
    }

    return {
        "answer": final_answer,
        "sub_questions": sub_questions,
        "retrieved_triples": final_triples,
        "retrieved_chunks": final_chunk_contents,
        "reasoning_steps": reasoning_steps,
        "visualization_data": visualization_data,
        "token_usage": getattr(kt_retriever, "get_token_usage", lambda: {"total_tokens": 0})(),
    }


    # visualization helpers moved to app.utils.visualization


def _is_graph_empty(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return True
        with open(path, "r", encoding="utf-8") as f:
            data = __import__("json").load(f)
        if isinstance(data, list):
            return len(data) == 0
        if isinstance(data, dict):
            nodes = data.get("nodes") or []
            edges = data.get("edges") or data.get("links") or []
            return (len(nodes) == 0) and (len(edges) == 0)
        return True
    except Exception:
        return True


def _get_schema_path_for_dataset(dataset_name: str) -> str:
    from app.services.dataset_service import get_schema_path_for_dataset
    return get_schema_path_for_dataset(dataset_name)
