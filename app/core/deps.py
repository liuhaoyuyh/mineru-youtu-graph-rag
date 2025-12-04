from fastapi import Depends
from app.core.ws import get_manager, ConnectionManager
from app.core.config import get_app_config


def ws_manager_dep() -> ConnectionManager:
    return get_manager()


def config_dep():
    return get_app_config()

