from pydantic import BaseModel
from typing import Optional, Dict


class GraphConstructionRequest(BaseModel):
    dataset_name: str


class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None

