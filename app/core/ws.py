import json
from datetime import datetime
from typing import Dict
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        ws = self.active_connections.get(client_id)
        if not ws:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            self.disconnect(client_id)


_manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    return _manager


async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    await _manager.send_message(
        {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        },
        client_id,
    )

