"""
思维导图功能模块 - 从 backend.py 迁移
"""
import os
import re
import json
import ast
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from fastapi import HTTPException, UploadFile, File
from pydantic import BaseModel

from utils.logger import logger, setup_logger
from app.core.graphrag import GRAPHRAG_AVAILABLE, get_config, reload_config, retriever, decomposer
from app.core.ws import get_manager, send_progress_update
from app.services.dataset_service import get_schema_path_for_dataset


def _is_graph_empty(path: str) -> bool:
    """Check if graph file is empty"""
    try:
        if not os.path.exists(path):
            return True
        with open(path, 'r', encoding='utf-8') as f:
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


def _count_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        s = (text or "")
        return max(1, int(len(s) / 4))


def _setup_mindmap_logger():
    """Setup mindmap logger"""
    try:
        logs_dir = "output/logs"
        os.makedirs(logs_dir, exist_ok=True)
        fh_paths = [getattr(h, 'baseFilename', None) for h in logger.handlers]
        if not any(p and p.endswith("mindmap.log") for p in fh_paths):
            setup_logger(name="youtu-graphrag", level=logging.INFO, log_file=os.path.join(logs_dir, "mindmap.log"))
    except Exception:
        pass


# Request/Response models
class MindmapGenerateRequest(BaseModel):
    dataset_name: str
    materialize: Optional[bool] = False


class MindmapGenerateResponse(BaseModel):
    success: bool
    message: str
    dataset_name: str
    mindmap: Dict
    save_path: Optional[str] = None


class MindmapMaterializeRequest(BaseModel):
    dataset_name: str


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


# 思维导图功能函数
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

# Removed - convert_graphrag_format and convert_standard_format moved to app/utils/visualization.py

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

# Removed - ask_question function moved to retrieval_service
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

# Removed - functions moved to dataset_service (get_datasets, upload_schema, delete_dataset)

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
        if not os.path.exists(graph_path) or _is_graph_empty(graph_path):
            graph_path = "output/graphs/demo_new.json"
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")
        config = get_config("config/base_config.yaml")
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
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "start", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        await loop.run_in_executor(None, kt_retriever.build_indices)
        try:
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "indices_built", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "retrieval_start", "dataset": dataset_name, "module": name, "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "retrieval_done", "dataset": dataset_name, "module": name, "triples_count": len(triples), "chunks_count": len(contents), "triples_preview": triples[:5], "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "prompt_built", "dataset": dataset_name, "module": name, "prompt_type": prompt_type, "prompt_preview": (template or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            try:
                text = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(template))
            except Exception as e:
                text = f"Failed to generate explanation: {e}"
            try:
                logger.info(f"mindmap qa llm done dataset='{dataset_name}' module='{name}' tlen={len(text)}")
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "llm_done", "dataset": dataset_name, "module": name, "answer_preview": (text or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "module_complete", "dataset": dataset_name, "module": name, "saved": fpath, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
        full_md = os.path.join(save_dir, "full.md")
        with open(full_md, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(md_lines))
        saved.append(full_md)
        try:
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "complete", "dataset": dataset_name, "modules": len(nodes), "timestamp": datetime.now().isoformat()}, client_id)
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
        config = get_config("config/base_config.yaml")
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
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "start", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        await loop.run_in_executor(None, kt_retriever.build_indices)
        try:
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "indices_built", "dataset": dataset_name, "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "retrieval_start", "dataset": dataset_name, "module": name, "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "retrieval_done", "dataset": dataset_name, "module": name, "triples_count": len(triples), "chunks_count": len(contents), "triples_preview": triples[:5], "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "prompt_built", "dataset": dataset_name, "module": name, "prompt_type": prompt_type, "prompt_preview": (template or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
            try:
                text = await loop.run_in_executor(None, lambda: kt_retriever.generate_answer(template))
            except Exception as e:
                text = f"Failed to generate explanation: {e}"
            try:
                logger.info(f"mindmap qa md llm done dataset='{dataset_name}' module='{name}' tlen={len(text)}")
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "llm_done", "dataset": dataset_name, "module": name, "answer_preview": (text or "")[:300], "timestamp": datetime.now().isoformat()}, client_id)
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
                await get_manager().send_message({"type": "mindmap_qa_update", "stage": "module_complete", "dataset": dataset_name, "module": name, "saved": fpath, "timestamp": datetime.now().isoformat()}, client_id)
            except Exception:
                pass
        full_md = os.path.join(out_dir, "full.md")
        with open(full_md, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(md_lines))
        saved.append(full_md)
        try:
            await get_manager().send_message({"type": "mindmap_qa_update", "stage": "complete", "dataset": dataset_name, "modules": len(modules), "timestamp": datetime.now().isoformat()}, client_id)
        except Exception:
            pass
        return MindmapQAMdResponse(success=True, saved_files=saved, stats={"modules": len(modules), "total_triples": total_triples, "total_chunks": total_chunks})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
