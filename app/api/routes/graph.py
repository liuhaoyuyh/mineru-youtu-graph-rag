from fastapi import APIRouter, HTTPException
import os
from app.services.graph_service import construct_graph as svc_construct_graph, prepare_graph_visualization
from app.schemas.graph import GraphConstructionRequest, GraphConstructionResponse

router = APIRouter()



@router.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(request: GraphConstructionRequest, client_id: str = "default"):
    data = await svc_construct_graph(request.dataset_name, client_id)
    return GraphConstructionResponse(success=True, message="Knowledge graph constructed successfully", graph_data=data)


@router.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    graph_path = f"output/graphs/{dataset_name}_new.json"
    if not os.path.exists(graph_path):
        return {
            "nodes": [
                {"id": "node1", "name": "Example Entity 1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1},
        }
    return await prepare_graph_visualization(graph_path)
