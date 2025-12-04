import os
import json
import time
import sys

try:
    import requests
except Exception:
    print("Please install requests: pip install requests")
    sys.exit(1)


BASE_URL = os.environ.get("YOUTU_BASE_URL", "http://localhost:8000")


def _get(path):
    r = requests.get(BASE_URL + path, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path, json_body):
    r = requests.post(BASE_URL + path, json=json_body, timeout=60)
    r.raise_for_status()
    return r.json()


def prepare_dataset(dataset_name: str):
    base = os.path.join("data", "uploaded", dataset_name)
    os.makedirs(base, exist_ok=True)
    corpus_path = os.path.join(base, "corpus.json")
    if not os.path.exists(corpus_path):
        docs = [
            {"title": "简介", "text": "这是一个用于演示的文档，用于构建知识图谱与测试思维导图QA。"},
            {"title": "模块1", "text": "模块1介绍与关键点。"},
            {"title": "模块1.1", "text": "模块1.1的细节说明与步骤。"},
        ]
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    out = os.path.join(base, "mineru_out")
    os.makedirs(out, exist_ok=True)
    md1 = os.path.join(out, "module_01.md")
    md2 = os.path.join(out, "module_02.md")
    if not os.path.exists(md1):
        m1 = """# 模块一
## 1 概述
模块一的整体介绍与关键关系。
### 1.1 要点
列出主要要点。
"""
        with open(md1, "w", encoding="utf-8") as f:
            f.write(m1)
    if not os.path.exists(md2):
        m2 = """# 模块二
## 1 概述
模块二的整体介绍与关键关系。
### 1.1 步骤
列出主要步骤。
"""
        with open(md2, "w", encoding="utf-8") as f:
            f.write(m2)
    return corpus_path, md1


def wait_for_server():
    for _ in range(30):
        try:
            info = _get("/api/status")
            if info.get("status") == "ok":
                return True
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server not responding at /api/status")


def main():
    dataset = os.environ.get("YOUTU_DATASET", "YouTu-GraphRag")
    wait_for_server()
    prepare_dataset(dataset)
    print("[1] Construct graph...")
    _post("/api/construct-graph", {"dataset_name": dataset})
    print("[2] Generate mindmap...")
    mm = _post("/api/mindmap/generate", {"dataset_name": dataset, "materialize": True})
    print("mindmap saved:", mm.get("save_path"))
    print("[3] Mindmap QA...")
    qa = _post("/api/mindmap/qa", {"dataset_name": dataset, "save": True, "limit_per_module": 10})
    print("Saved files:")
    for p in qa.get("saved_files", []):
        print(" -", p)
    print("Stats:", qa.get("stats"))


if __name__ == "__main__":
    main()
