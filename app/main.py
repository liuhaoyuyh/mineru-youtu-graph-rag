from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

    from .api.routes import status, upload, datasets, graph, qa, mindmap, visualization, ws

    app.include_router(status.router)
    app.include_router(upload.router)
    app.include_router(datasets.router)
    app.include_router(graph.router)
    app.include_router(qa.router)
    app.include_router(mindmap.router)
    app.include_router(visualization.router)
    app.include_router(ws.router)

    from .core.events import register_events
    register_events(app)

    return app
