"""
文件上传服务
"""
import os
import json
import logging
from typing import List
from fastapi import UploadFile, HTTPException

from utils.logger import logger, setup_logger
from utils.mineru_adapter import parse_with_mineru
from app.core.ws import send_progress_update
from app.schemas.upload import FileUploadResponse


async def upload_files(files: List[UploadFile], client_id: str = "default") -> FileUploadResponse:
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
        # TODO 这里逻辑需要修改，如果目录存在，只需要检查里面的文件是否已经mineru构建完成，不需要重新使用mineru解析
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
                # Try MinerU to parse PDFs/images into text corpus
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
        
        # Save corpus data
        corpus_path = f"{upload_dir}/corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        # Ensure default demo schema exists
        from app.services.dataset_service import ensure_demo_schema_exists
        ensure_demo_schema_exists()
        
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



