import os
import json
import glob
import shutil
import subprocess
from typing import List, Dict, Any


def _get_mineru_bin() -> str:
    """Resolve MinerU executable path, supporting conda installs and env override.

    Order:
    - env MINERU_BIN
    - PATH `mineru`
    - conda prefix `CONDA_PREFIX/bin/mineru`
    - common conda path `/opt/anaconda3/envs/youtu-graphrag/bin/mineru`
    """
    env_bin = os.getenv("MINERU_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    which_bin = shutil.which("mineru")
    if which_bin:
        return which_bin
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "mineru")
        if os.path.isfile(candidate):
            return candidate
    # Fallback: common macOS conda path
    fallback = "/opt/anaconda3/envs/youtu-graphrag/bin/mineru"
    if os.path.isfile(fallback):
        return fallback
    return ""


def is_mineru_available() -> bool:
    """Check if MinerU CLI is available."""
    return bool(_get_mineru_bin())


def _build_mineru_runner() -> List[str]:
    """Build the mineru invocation prefix.

    Prefer `conda run -n <env> mineru` to ensure correct Python deps,
    fallback to direct mineru binary.
    Env name can be overridden via MINERU_CONDA_ENV (default: youtu-graphrag).
    """
    env_name = os.getenv("MINERU_CONDA_ENV", "youtu-graphrag")
    # If explicitly requested or conda exists, use conda run
    use_conda = os.getenv("MINERU_RUNNER", "conda") == "conda"
    conda_bin = shutil.which("conda") or os.path.join(os.getenv("CONDA_PREFIX", ""), "bin", "conda")
    if use_conda and conda_bin and os.path.isfile(conda_bin):
        return [conda_bin, "run", "-n", env_name, "mineru"]
    mineru_bin = _get_mineru_bin()
    return [mineru_bin] if mineru_bin else []


def parse_with_mineru(input_path: str, output_dir: str) -> List[Dict]:
    """
    Use MinerU CLI to parse a document (PDF/image) into text corpus entries.

    Returns a list of corpus items: [{"title": <file>, "text": <content>}]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Run MinerU CLI: prefer OCR for robust text extraction on images
    # Prefer local model source; allow device override via env (default cpu)
    device = os.getenv("MINERU_DEVICE", "cpu")
    runner = _build_mineru_runner()
    if not runner:
        # No available mineru executable
        return []

    # Prepare environment to prefer local caches
    env = os.environ.copy()
    local_cache = env.get("MINERU_LOCAL_CACHE") or os.path.abspath("models/hf_cache")
    # Set huggingface cache hints for local source
    env.setdefault("HUGGINGFACE_HUB_CACHE", local_cache)
    env.setdefault("HF_HOME", local_cache)
    force_ocr = os.getenv("MINERU_FORCE_OCR", "1") in ("1", "true", "True")
    if force_ocr:
        ocr_cmd = [
            *runner,
            "-p", input_path,
            "-o", output_dir,
            "-m", "ocr",
            "--source", "local",
            "-d", device,
        ]
        proc = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU OCR failed: stderr='{(proc.stderr or '').strip()}' stdout='{(proc.stdout or '').strip()}'")
    else:
        pipeline_cmd = [
            *runner,
            "-p", input_path,
            "-o", output_dir,
            "-m", "auto",
            "-b", "pipeline",
            "--source", "local",
            "-d", device,
        ]
        proc = subprocess.run(pipeline_cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            # Try OCR as fallback
            ocr_cmd = [
                *runner,
                "-p", input_path,
                "-o", output_dir,
                "-m", "ocr",
                "--source", "local",
                "-d", device,
            ]
            proc_ocr = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
            if proc_ocr.returncode != 0:
                raise RuntimeError(
                    f"MinerU failed: pipeline stderr='{(proc.stderr or '').strip()}', "
                    f"ocr stderr='{(proc_ocr.stderr or '').strip()}'"
                )

    # Find output JSON produced by MinerU (path can vary by backend/version)
    # Try a set of common filenames and patterns
    search_patterns = [
        os.path.join(output_dir, "**", "content_list.json"),
        os.path.join(output_dir, "**", "middle.json"),
        os.path.join(output_dir, "**", "content.json"),
        os.path.join(output_dir, "**", "content_txt.json"),
        os.path.join(output_dir, "**", "layout.json"),
        os.path.join(output_dir, "**", "*content*.json"),
        # Fallback: include all JSON outputs to maximize text extraction
        os.path.join(output_dir, "**", "*.json"),
    ]
    result_files: List[str] = []
    for pattern in search_patterns:
        result_files.extend(glob.glob(pattern, recursive=True))
    # Deduplicate while preserving order
    seen = set()
    result_files = [f for f in result_files if not (f in seen or seen.add(f))]

    # If JSON not found, try to aggregate .txt/.md outputs as a fallback
    if not result_files:
        text_like_files = []
        text_like_files.extend(glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True))
        text_like_files.extend(glob.glob(os.path.join(output_dir, "**", "*.md"), recursive=True))
        texts: List[str] = []
        for tf in text_like_files:
            try:
                with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read().strip()
                    if t:
                        texts.append(t)
            except Exception:
                # Ignore unreadable files
                pass
        content = "\n\n".join(texts) if texts else ""
        title = os.path.basename(input_path)
        # Return empty when no content; upstream will gracefully fallback
        return [{"title": title, "text": content}] if content else []

    # Helper: recursively extract text-like strings from arbitrary JSON structures
    def _extract_texts_from_json(obj: Any) -> List[str]:
        collected: List[str] = []

        def walk(node: Any):
            if isinstance(node, dict):
                for k, v in node.items():
                    # Direct text fields
                    if k in ("text", "ocr_text", "content", "caption") and isinstance(v, str):
                        s = v.strip()
                        if s:
                            collected.append(s)
                    # Arrays of strings under common caption keys
                    if k in ("image_caption", "captions", "footnotes", "image_footnote") and isinstance(v, list):
                        for itm in v:
                            if isinstance(itm, str):
                                s = itm.strip()
                                if s:
                                    collected.append(s)
                            else:
                                walk(itm)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for itm in node:
                    walk(itm)
            # Ignore other primitive types

        walk(obj)
        return collected

    # Aggregate texts across all discovered JSON outputs
    texts: List[str] = []
    for jf in result_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts.extend(_extract_texts_from_json(data))
        except Exception:
            # Skip unreadable/bad json
            continue

    content = "\n\n".join(texts) if texts else ""
    title = os.path.basename(input_path)
    return [{"title": title, "text": content}] if content else []