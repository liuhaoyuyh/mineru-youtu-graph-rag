import os
import json
import glob
import shutil
import subprocess
from typing import List, Dict, Any
from utils.logger import logger
from config import get_config


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
    """Check if MinerU can be invoked (via conda-run or direct binary)."""
    runner = _build_mineru_runner()
    return bool(runner)


def _build_mineru_runner() -> List[str]:
    """Build the mineru invocation prefix.

    Prefer `conda run -n <env> mineru` to ensure correct Python deps,
    fallback to direct mineru binary.
    Env name can be overridden via MINERU_CONDA_ENV (default: youtu-graphrag).
    """
    env_name = os.getenv("MINERU_CONDA_ENV", "youtu-graphrag")
    use_conda = os.getenv("MINERU_RUNNER", "conda") == "conda"
    conda_bin = shutil.which("conda") or os.path.join(os.getenv("CONDA_PREFIX", ""), "bin", "conda")
    if use_conda and conda_bin and os.path.isfile(conda_bin):
        runner = [conda_bin, "run", "-n", env_name, "mineru"]
        logger.info(f"mineru runner: {' '.join(runner)}")
        return runner
    mineru_bin = _get_mineru_bin()
    if mineru_bin:
        runner = [mineru_bin]
        logger.info(f"mineru runner: {' '.join(runner)}")
        return runner
    logger.warning("mineru runner not resolved")
    return []


def parse_with_mineru(input_path: str, output_dir: str) -> List[Dict]:
    """
    Use MinerU CLI to parse a document (PDF/image) into text corpus entries.

    Returns a list of corpus items: [{"title": <file>, "text": <content>}]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve configuration (file > env > default)
    cfg = get_config("config/base_config.yaml")
    mineru_cfg = (cfg.config_data or {}).get("mineru", {})
    device = mineru_cfg.get("device") or os.getenv("MINERU_DEVICE", "cpu")
    runner = _build_mineru_runner()
    logger.info(f"mineru parse start: input='{input_path}' output='{output_dir}' device='{device}'")
    if not runner:
        # No available mineru executable
        return []

    # Prepare environment to prefer local caches
    env = os.environ.copy()
    local_cache = env.get("MINERU_LOCAL_CACHE") or os.path.abspath("models/hf_cache")
    # Set huggingface cache hints for local source
    env.setdefault("HUGGINGFACE_HUB_CACHE", local_cache)
    env.setdefault("HF_HOME", local_cache)
    # Configuration-driven overrides
    source = mineru_cfg.get("source") or os.getenv("MINERU_SOURCE", "huggingface")
    mode = str(mineru_cfg.get("mode", "auto")).strip() or "auto"
    backend = str(mineru_cfg.get("backend", "pipeline")).strip() or "pipeline"
    initial_backend = backend
    base_pdf_name = os.path.splitext(os.path.basename(input_path))[0]
    subdir = "ocr" if (mode == "ocr" or backend == "pipeline") else "vlm"
    target_root = os.path.join(output_dir, base_pdf_name, subdir)
    if "force_ocr" in mineru_cfg:
        force_ocr = bool(mineru_cfg.get("force_ocr"))
    else:
        force_ocr = os.getenv("MINERU_FORCE_OCR", "1") in ("1", "true", "True")

    # Decide command path
    if force_ocr or mode == "ocr":
        ocr_cmd = [
            *runner,
            "-p", input_path,
            "-o", output_dir,
            "-m", "ocr",
            "--source", source,
            "-d", device,
        ]
        logger.info(f"mineru cmd: {' '.join(ocr_cmd)}")
        proc = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
        logger.info(f"mineru returncode={proc.returncode} stdout_len={len(proc.stdout or '')} stderr_len={len(proc.stderr or '')}")
        if proc.stderr:
            _stderr_head = (proc.stderr or "")[:500].replace("\n", " ")
            logger.info(f"mineru stderr head: {_stderr_head}")
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU OCR failed: stderr='{(proc.stderr or '').strip()}' stdout='{(proc.stdout or '').strip()}'")
    else:
        ran_pipeline_fallback = False
        pipeline_cmd = [
            *runner,
            "-p", input_path,
            "-o", output_dir,
            "-m", mode,
            "-b", backend,
            "--source", source,
            "-d", device,
        ]
        logger.info(f"mineru cmd: {' '.join(pipeline_cmd)}")
        proc = subprocess.run(pipeline_cmd, capture_output=True, text=True, env=env)
        logger.info(f"mineru returncode={proc.returncode} stdout_len={len(proc.stdout or '')} stderr_len={len(proc.stderr or '')}")
        if proc.stderr:
            _stderr_head = (proc.stderr or "")[:500].replace("\n", " ")
            logger.info(f"mineru stderr head: {_stderr_head}")
        if proc.returncode != 0:
            if initial_backend != "pipeline":
                pipeline_fallback_cmd = [
                    *runner,
                    "-p", input_path,
                    "-o", output_dir,
                    "-m", mode,
                    "-b", "pipeline",
                    "--source", source,
                    "-d", device,
                ]
                logger.info(f"mineru pipeline-fallback cmd: {' '.join(pipeline_fallback_cmd)}")
                proc_pf = subprocess.run(pipeline_fallback_cmd, capture_output=True, text=True, env=env)
                logger.info(f"mineru pipeline-fallback returncode={proc_pf.returncode} stdout_len={len(proc_pf.stdout or '')} stderr_len={len(proc_pf.stderr or '')}")
                if proc_pf.stderr:
                    _stderr_head_pf = (proc_pf.stderr or "")[:500].replace("\n", " ")
                    logger.info(f"mineru pipeline-fallback stderr head: {_stderr_head_pf}")
                ran_pipeline_fallback = True
                if proc_pf.returncode == 0:
                    pass
                else:
                    ocr_cmd = [
                        *runner,
                        "-p", input_path,
                        "-o", output_dir,
                        "-m", "ocr",
                        "--source", source,
                        "-d", device,
                    ]
                    logger.info(f"mineru fallback cmd: {' '.join(ocr_cmd)}")
                    proc_ocr = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
                    logger.info(f"mineru fallback returncode={proc_ocr.returncode} stdout_len={len(proc_ocr.stdout or '')} stderr_len={len(proc_ocr.stderr or '')}")
                    if proc_ocr.stderr:
                        _stderr_head_fb = (proc_ocr.stderr or "")[:500].replace("\n", " ")
                        logger.info(f"mineru fallback stderr head: {_stderr_head_fb}")
                    if proc_ocr.returncode != 0:
                        raise RuntimeError(
                            f"MinerU failed: backend stderr='{(proc.stderr or '').strip()}', "
                            f"pipeline stderr='{(proc_pf.stderr or '').strip()}', "
                            f"ocr stderr='{(proc_ocr.stderr or '').strip()}'"
                        )
            else:
                ocr_cmd = [
                    *runner,
                    "-p", input_path,
                    "-o", output_dir,
                    "-m", "ocr",
                    "--source", source,
                    "-d", device,
                ]
                logger.info(f"mineru fallback cmd: {' '.join(ocr_cmd)}")
                proc_ocr = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
                logger.info(f"mineru fallback returncode={proc_ocr.returncode} stdout_len={len(proc_ocr.stdout or '')} stderr_len={len(proc_ocr.stderr or '')}")
                if proc_ocr.stderr:
                    _stderr_head_fb = (proc_ocr.stderr or "")[:500].replace("\n", " ")
                    logger.info(f"mineru fallback stderr head: {_stderr_head_fb}")
                if proc_ocr.returncode != 0:
                    raise RuntimeError(
                        f"MinerU failed: backend stderr='{(proc.stderr or '').strip()}', "
                        f"ocr stderr='{(proc_ocr.stderr or '').strip()}'"
                    )
            ocr_cmd = [
                *runner,
                "-p", input_path,
                "-o", output_dir,
                "-m", "ocr",
                "--source", source,
                "-d", device,
            ]
            logger.info(f"mineru fallback cmd: {' '.join(ocr_cmd)}")
            proc_ocr = subprocess.run(ocr_cmd, capture_output=True, text=True, env=env)
            logger.info(f"mineru fallback returncode={proc_ocr.returncode} stdout_len={len(proc_ocr.stdout or '')} stderr_len={len(proc_ocr.stderr or '')}")
            if proc_ocr.stderr:
                _stderr_head_fb = (proc_ocr.stderr or "")[:500].replace("\n", " ")
                logger.info(f"mineru fallback stderr head: {_stderr_head_fb}")
            if proc_ocr.returncode != 0:
                raise RuntimeError(
                    f"MinerU failed: pipeline stderr='{(proc.stderr or '').strip()}', "
                    f"ocr stderr='{(proc_ocr.stderr or '').strip()}'"
                )

    # Find output JSON produced by MinerU (path can vary by backend/version)
    # Try a set of common filenames and patterns
    search_patterns = [
        os.path.join(target_root, "**", "content_list.json"),
        os.path.join(target_root, "**", "middle.json"),
        os.path.join(target_root, "**", "content.json"),
        os.path.join(target_root, "**", "content_txt.json"),
        os.path.join(target_root, "**", "layout.json"),
        os.path.join(target_root, "**", "*content*.json"),
        os.path.join(output_dir, "**", "content_list.json"),
        os.path.join(output_dir, "**", "*.json"),
    ]
    result_files: List[str] = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        logger.info(f"mineru glob pattern '{pattern}' -> {len(files)}")
        result_files.extend(files)
    # Deduplicate while preserving order
    seen = set()
    result_files = [f for f in result_files if not (f in seen or seen.add(f))]

    # If JSON not found, try to aggregate .txt/.md outputs as a fallback
    logger.info(f"mineru outputs found: {len(result_files)} files")
    if not result_files:
        try:
            tree = glob.glob(os.path.join(output_dir, "**"), recursive=True)
            logger.info(f"mineru output tree entries={len(tree)}")
            sample = ", ".join(sorted(tree)[:10])
            logger.info(f"mineru output tree sample: {sample}")
        except Exception:
            pass
        if initial_backend != "pipeline" and not ran_pipeline_fallback:
            try:
                pipeline_fallback_cmd2 = [
                    *runner,
                    "-p", input_path,
                    "-o", output_dir,
                    "-m", mode,
                    "-b", "pipeline",
                    "--source", source,
                    "-d", device,
                ]
                logger.info(f"mineru empty-output pipeline-fallback cmd: {' '.join(pipeline_fallback_cmd2)}")
                proc_pf2 = subprocess.run(pipeline_fallback_cmd2, capture_output=True, text=True, env=env)
                logger.info(f"mineru empty-output pipeline-fallback returncode={proc_pf2.returncode} stdout_len={len(proc_pf2.stdout or '')} stderr_len={len(proc_pf2.stderr or '')}")
                if proc_pf2.stderr:
                    _stderr_head_pf2 = (proc_pf2.stderr or "")[:500].replace("\n", " ")
                    logger.info(f"mineru empty-output pipeline-fallback stderr head: {_stderr_head_pf2}")
            except Exception:
                pass
            for pattern in search_patterns:
                files = glob.glob(pattern, recursive=True)
                result_files.extend(files)
            seen2a = set()
            result_files = [f for f in result_files if not (f in seen2a or seen2a.add(f))]
        try:
            ocr_cmd2 = [
                *runner,
                "-p", input_path,
                "-o", output_dir,
                "-m", "ocr",
                "--source", source,
                "-d", device,
            ]
            logger.info(f"mineru empty-output fallback cmd: {' '.join(ocr_cmd2)}")
            proc2 = subprocess.run(ocr_cmd2, capture_output=True, text=True, env=env)
            logger.info(f"mineru empty-output fallback returncode={proc2.returncode} stdout_len={len(proc2.stdout or '')} stderr_len={len(proc2.stderr or '')}")
            if proc2.stderr:
                _stderr_head2 = (proc2.stderr or "")[:500].replace("\n", " ")
                logger.info(f"mineru empty-output fallback stderr head: {_stderr_head2}")
        except Exception:
            pass
        if not result_files:
            for pattern in search_patterns:
                files = glob.glob(pattern, recursive=True)
                result_files.extend(files)
            seen2 = set()
            result_files = [f for f in result_files if not (f in seen2 or seen2.add(f))]
        image_patterns = [
            os.path.join(target_root, "**", "images", "**", "*.png"),
            os.path.join(target_root, "**", "images", "**", "*.jpg"),
            os.path.join(target_root, "**", "images", "**", "*.jpeg"),
        ]
        images: List[str] = []
        for pat in image_patterns:
            images.extend(glob.glob(pat, recursive=True))
        if images:
            items = []
            for idx, ip in enumerate(sorted(images)):
                items.append({"type": "image", "img_path": ip, "page_idx": idx})
            try:
                os.makedirs(target_root, exist_ok=True)
                out_json = os.path.join(target_root, "content_list.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False)
                result_files.append(out_json)
                logger.info(f"mineru synthesized content_list.json with {len(items)} images")
            except Exception:
                pass
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
                    pass
            content = "\n\n".join(texts) if texts else ""
            title = os.path.basename(input_path)
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
    logger.info(f"mineru aggregated texts: {len(texts)} entries, content_len={len(content)}")
    title = os.path.basename(input_path)
    return [{"title": title, "text": content}] if content else []