#!/usr/bin/env python3
"""
Youtu-GraphRAG Backend Server Entry Point
"""
import uvicorn
from app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
