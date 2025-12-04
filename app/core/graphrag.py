"""
GraphRAG 组件可用性检查和导入
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import logger

# Import constructor first (does not depend on FAISS)
try:
    from models.constructor import kt_gen as constructor
    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG constructor loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    constructor = None
    logger.error(f"⚠️  GraphRAG constructor not available: {e}")

# Import retriever components optionally (may depend on FAISS)
RETRIEVER_AVAILABLE = False
decomposer = None
retriever = None
try:
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
    RETRIEVER_AVAILABLE = True
    logger.info("✅ GraphRAG retriever components loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  GraphRAG retriever components not available: {e}")

# Import config functions separately (always available)
try:
    from config import get_config, ConfigManager, reload_config
except ImportError:
    get_config = None
    ConfigManager = None
    reload_config = None

__all__ = [
    "GRAPHRAG_AVAILABLE",
    "RETRIEVER_AVAILABLE",
    "constructor",
    "decomposer",
    "retriever",
    "get_config",
    "ConfigManager",
    "reload_config",
]
