from utils import logger
import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any

from config import get_config
from utils.call_llm_api import LLMCompletionCall
from utils.call_llm_api import call_vlm

PROMPTS = {}

PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = "You are an expert image analyst. Provide detailed, accurate descriptions."
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = "You are an expert data analyst. Provide detailed table analysis with specific insights."
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = "You are an expert mathematician. Provide detailed mathematical analysis."

PROMPTS["vision_prompt"] = """Please analyze this image in detail and provide a detailed description.\n\nImage Path: {image_path}\nCaptions: {captions}\nFootnotes: {footnotes}\n"""
PROMPTS["vision_prompt_with_context"] = """Please analyze this image in detail, considering the surrounding context, and provide a detailed description.\n\nContext:\n{context}\n\nImage Path: {image_path}\nCaptions: {captions}\nFootnotes: {footnotes}\n"""
PROMPTS["table_prompt"] = """Please analyze this table content and provide a detailed description.\n\nImage Path: {table_img_path}\nCaption: {table_caption}\nBody: {table_body}\nFootnotes: {table_footnote}\n"""
PROMPTS["table_prompt_with_context"] = """Please analyze this table content considering the surrounding context and provide a detailed description.\n\nContext:\n{context}\n\nImage Path: {table_img_path}\nCaption: {table_caption}\nBody: {table_body}\nFootnotes: {table_footnote}\n"""
PROMPTS["equation_prompt"] = """Please analyze this mathematical equation and provide a detailed description.\n\nEquation: {equation_text}\nFormat: {equation_format}\n"""
PROMPTS["equation_prompt_with_context"] = """Please analyze this mathematical equation considering the surrounding context and provide a detailed description.\n\nContext:\n{context}\n\nEquation: {equation_text}\nFormat: {equation_format}\n"""

PROMPTS["image_chunk"] = "\nImage Content Analysis:\nImage Path: {image_path}\nCaptions: {captions}\nFootnotes: {footnotes}\n\nVisual Analysis: {enhanced_caption}\n"
PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""
PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""
PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""


class ContextExtractor:
    def __init__(self, window: int = 1):
        self.window = int(window or 1)

    def extract(self, content_list: List[Dict[str, Any]], current_item: Dict[str, Any]) -> str:
        if not content_list:
            return ""
        current_page = int(current_item.get("page_idx", 0))
        start_page = max(0, current_page - self.window)
        end_page = current_page + self.window + 1
        texts = []
        for it in content_list:
            if str(it.get("type", "")) == "text":
                p = int(it.get("page_idx", 0))
                if start_page <= p < end_page:
                    val = it.get("text", "")
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
        return "\n".join(texts)


def _encode_image_to_base64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def _parse_response_json(resp: str) -> Dict[str, Any]:
    try:
        return json.loads(resp)
    except Exception:
        pass
    import re
    m = re.search(r"\{[\s\S]*\}", resp)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"detailed_description": resp.strip(), "entity_info": {}}
    return {"detailed_description": resp.strip(), "entity_info": {}}


def _fix_paths(content_list: List[Dict[str, Any]], base_dir: Path) -> None:
    for item in content_list:
        if isinstance(item, dict):
            for fn in ["img_path", "table_img_path", "equation_img_path"]:
                if fn in item and item[fn]:
                    p = Path(item[fn])
                    if not p.is_absolute():
                        item[fn] = str((base_dir / p).resolve())


def _apply_chunk_template(content_type: str, item: Dict[str, Any], description: str) -> str:
    if content_type == "image":
        image_path = item.get("img_path", "")
        captions = item.get("image_caption", item.get("img_caption", []))
        footnotes = item.get("image_footnote", item.get("img_footnote", []))
        return PROMPTS["image_chunk"].format(
            image_path=image_path,
            captions=", ".join(captions) if captions else "None",
            footnotes=", ".join(footnotes) if footnotes else "None",
            enhanced_caption=description,
        )
    if content_type == "table":
        table_img_path = item.get("img_path", "")
        table_caption = item.get("table_caption", [])
        table_body = item.get("table_body", "")
        table_footnote = item.get("table_footnote", [])
        return PROMPTS["table_chunk"].format(
            table_img_path=table_img_path,
            table_caption=", ".join(table_caption) if table_caption else "None",
            table_body=table_body,
            table_footnote=", ".join(table_footnote) if table_footnote else "None",
            enhanced_caption=description,
        )
    if content_type == "equation":
        equation_text = item.get("text", "")
        equation_format = item.get("text_format", "")
        return PROMPTS["equation_chunk"].format(
            equation_text=equation_text,
            equation_format=equation_format,
            enhanced_caption=description,
        )
    return PROMPTS["generic_chunk"].format(
        content_type=content_type.title(),
        content=str(item),
        enhanced_caption=description,
    )


def _analyze_image(item: Dict[str, Any], context: str, llm: LLMCompletionCall, use_vlm: bool) -> str:
    image_path = item.get("img_path", "")
    captions = item.get("image_caption", item.get("img_caption", []))
    footnotes = item.get("image_footnote", item.get("img_footnote", []))
    if context:
        prompt = PROMPTS["vision_prompt_with_context"].format(
            context=context,
            image_path=image_path,
            captions=", ".join(captions) if captions else "None",
            footnotes=", ".join(footnotes) if footnotes else "None",
        )
    else:
        prompt = PROMPTS["vision_prompt"].format(
            image_path=image_path,
            captions=", ".join(captions) if captions else "None",
            footnotes=", ".join(footnotes) if footnotes else "None",
        )
    if use_vlm and image_path:
        img64 = _encode_image_to_base64(image_path)
        if img64:
            resp = call_vlm(prompt, img64, system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"])
            data = _parse_response_json(resp)
            return str(data.get("detailed_description", resp)).strip()
    resp = llm.call_api(prompt)
    data = _parse_response_json(resp)
    return str(data.get("detailed_description", resp)).strip()


def _analyze_table(item: Dict[str, Any], context: str, llm: LLMCompletionCall) -> str:
    table_img_path = item.get("img_path", "")
    table_caption = item.get("table_caption", [])
    table_body = item.get("table_body", "")
    table_footnote = item.get("table_footnote", [])
    if context:
        prompt = PROMPTS["table_prompt_with_context"].format(
            context=context,
            table_img_path=table_img_path,
            table_caption=", ".join(table_caption) if table_caption else "None",
            table_body=table_body,
            table_footnote=", ".join(table_footnote) if table_footnote else "None",
        )
    else:
        prompt = PROMPTS["table_prompt"].format(
            table_img_path=table_img_path,
            table_caption=", ".join(table_caption) if table_caption else "None",
            table_body=table_body,
            table_footnote=", ".join(table_footnote) if table_footnote else "None",
        )
    resp = llm.call_api(prompt)
    data = _parse_response_json(resp)
    return str(data.get("detailed_description", resp)).strip()


def _analyze_equation(item: Dict[str, Any], context: str, llm: LLMCompletionCall) -> str:
    equation_text = item.get("text", "")
    equation_format = item.get("text_format", "")
    if context:
        prompt = PROMPTS["equation_prompt_with_context"].format(
            context=context,
            equation_text=equation_text,
            equation_format=equation_format,
        )
    else:
        prompt = PROMPTS["equation_prompt"].format(
            equation_text=equation_text,
            equation_format=equation_format,
        )
    resp = llm.call_api(prompt)
    data = _parse_response_json(resp)
    return str(data.get("detailed_description", resp)).strip()


def _find_content_list_file(mineru_out_dir: Path):
    patterns = [
        "**/*content_list.json",
        "**/*content*.json",
        "**/*_con",
        "**/*_con.json",
    ]
    for pat in patterns:
        for p in mineru_out_dir.glob(pat):
            if p.is_file():
                return p
    # Fallback: scan files to find a likely content list
    for p in mineru_out_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if ("content" in name or name.endswith("_con")):
            return p
    return None


from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_single_item(item, ctx, llm, vlm_enabled, dataset_name):
    t = item["type"]

    if t == "image":
        desc = _analyze_image(item, ctx, llm, vlm_enabled)
    elif t == "table":
        desc = _analyze_table(item, ctx, llm)
    else:
        desc = _analyze_equation(item, ctx, llm)

    chunk_text = _apply_chunk_template(t, item, desc)
    return {"title": dataset_name, "text": chunk_text}


def build_chunks_for_dataset(dataset_name: str) -> List[Dict[str, str]]:
    cfg = get_config("config/base_config.yaml")
    mm_cfg = (cfg.config_data or {}).get("multimodal", {})
    mineru_base = mm_cfg.get("mineru_out_base", "data/uploaded")
    mineru_out_dir = Path(mineru_base) / dataset_name / "mineru_out"
    if not mineru_out_dir.exists():
        return []

    content_file = _find_content_list_file(mineru_out_dir)
    if not content_file or not content_file.exists():
        return []

    # load content list
    try:
        raw = content_file.read_text(encoding="utf-8")
        content_list = json.loads(raw)
    except Exception:
        import json_repair
        content_list = json_repair.loads(raw)

    _fix_paths(content_list, content_file.parent)

    extractor = ContextExtractor(window=mm_cfg.get("context_window", 1))
    llm = LLMCompletionCall()
    vlm_enabled = bool(mm_cfg.get("vlm_enabled", False))
    chunks: List[Dict[str, str]] = []

    # only target types
    target_items = [
        (i, item) for i, item in enumerate(content_list)
        if str(item.get("type", "")) in ("image", "table", "equation")
    ]

    # precompute context
    try:
        all_ctx = extractor.precompute_context(content_list)
    except AttributeError:
        all_ctx = {id(item): extractor.extract(content_list, item)
                   for _, item in target_items}

    # -----------------------------
    # 🚀 SUPER SPEED: 线程池并行 LLM 推理
    # -----------------------------
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        for i, item in target_items:
            print(i)
            ctx = all_ctx.get(id(item))
            futures.append(
                executor.submit(
                    _process_single_item,
                    item, ctx, llm, vlm_enabled, dataset_name
                )
            )

        for f in as_completed(futures):
            chunks.append(f.result())

    return chunks
