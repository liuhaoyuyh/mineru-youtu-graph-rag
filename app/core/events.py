import os
from fastapi import FastAPI
from utils.logger import logger


def register_events(app: FastAPI) -> None:
    @app.on_event("startup")
    async def startup_event():
        os.makedirs("data/uploaded", exist_ok=True)
        os.makedirs("output/graphs", exist_ok=True)
        os.makedirs("output/logs", exist_ok=True)
        os.makedirs("schemas", exist_ok=True)
        logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")

