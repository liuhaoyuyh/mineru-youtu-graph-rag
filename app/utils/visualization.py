import ast
from typing import Dict, List


def convert_graphrag_format(graph_data: list) -> Dict:
    nodes_dict = {}
    links = []
    for item in graph_data:
        if not isinstance(item, dict):
            continue
        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")

        def _fallback_node_id(node: Dict) -> str:
            props = node.get("properties", {}) or {}
            name = props.get("name") or props.get("summary") or props.get("caption") or props.get("schema_type")
            if isinstance(name, (list, dict)):
                name = str(name)
            label = node.get("label", "entity")
            chunk_id = props.get("chunk id")
            candidate = name or (f"{label}_{chunk_id}" if chunk_id else label)
            return str(candidate) if candidate else label

        start_id = ""
        end_id = ""
        if start_node:
            start_id = _fallback_node_id(start_node)
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": str(start_id)[:30],
                    "category": start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {}),
                }
        if end_node:
            end_id = _fallback_node_id(end_node)
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": str(end_id)[:30],
                    "category": end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {}),
                }

        if start_id and end_id:
            links.append({"source": start_id, "target": end_id, "name": relation, "value": 1})

    categories_set = {node["category"] for node in nodes_dict.values()}
    categories = [{"name": cat_name, "itemStyle": {"color": f"hsl({i * 360 / len(categories_set)}, 70%, 60%)"}} for i, cat_name in enumerate(categories_set)]
    nodes = list(nodes_dict.values())
    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


def convert_standard_format(graph_data: Dict) -> Dict:
    nodes = []
    links = []
    categories = []
    node_types = {node.get("type", "entity") for node in graph_data.get("nodes", [])}
    for i, node_type in enumerate(node_types):
        categories.append({"name": node_type, "itemStyle": {"color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"}})
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": node.get("type", "entity"),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", []),
        })
    for edge in graph_data.get("edges", []):
        links.append({"source": edge.get("source", ""), "target": edge.get("target", ""), "name": edge.get("relation", "related_to"), "value": edge.get("weight", 1)})
    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


def prepare_subquery_visualization(sub_questions: List[Dict], reasoning_steps: List[Dict]) -> Dict:
    nodes = [{"id": "original", "name": "Original Question", "category": "question", "symbolSize": 40}]
    links = []
    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append({"id": sub_id, "name": sub_q.get("sub-question", "")[:20] + "...", "category": "sub_question", "symbolSize": 30})
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})
    return {"nodes": nodes, "links": links, "categories": [{"name": "question", "itemStyle": {"color": "#ff6b6b"}}, {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}}]}


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    nodes = []
    links = []
    node_set = set()
    for triple in triples[:10]:
        try:
            if triple.startswith("[") and triple.endswith("]"):
                try:
                    parts = ast.literal_eval(triple)
                except Exception:
                    continue
                if len(parts) == 3:
                    source, relation, target = parts
                    for entity in [source, target]:
                        if entity not in node_set:
                            node_set.add(entity)
                            nodes.append({"id": str(entity), "name": str(entity)[:20], "category": "entity", "symbolSize": 20})
                    links.append({"source": str(source), "target": str(target), "name": str(relation)})
        except Exception:
            continue
    return {"nodes": nodes, "links": links, "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}]}


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    steps_data = []
    for i, step in enumerate(reasoning_steps):
        steps_data.append({
            "step": i + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:50],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0),
        })
    return {"steps": steps_data, "timeline": [step["processing_time"] for step in steps_data]}

