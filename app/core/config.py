from typing import Optional
from config import get_config, reload_config


_config_cache: Optional[object] = None


def get_app_config() -> object:
    global _config_cache
    if _config_cache is None:
        _config_cache = get_config("config/base_config.yaml")
    return _config_cache


def reload_app_config() -> object:
    global _config_cache
    try:
        _config_cache = reload_config("config/base_config.yaml")
    except Exception:
        _config_cache = get_config("config/base_config.yaml")
    return _config_cache

