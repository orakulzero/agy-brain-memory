import pytest
from agy_brain_memory.knowledge_graph import TemporalEntityGraph, extract_temporal_knowledge_graph

def test_add_entity():
    graph = TemporalEntityGraph()
    node = graph.add_entity("node1", "Type1", {"prop": 1}, 10.0)
    assert "node1" in graph.entities
    assert graph.entities["node1"]["entity_type"] == "Type1"

def test_add_relation():
    graph = TemporalEntityGraph()
    rel = graph.add_relation("sub", "pred", "obj", 5.0, 10.0, 0.9)
    assert len(graph.relations) == 1
    assert "sub" in graph.entities
    assert "obj" in graph.entities
    assert graph.relations[0]["subject"] == "sub"

def test_query_point_in_time():
    graph = TemporalEntityGraph()
    graph.add_relation("A", "is", "B", valid_from=1.0, valid_to=5.0)
    graph.add_relation("A", "is", "C", valid_from=5.0, valid_to=10.0)
    
    res1 = graph.query_point_in_time(3.0)
    assert len(res1) == 1
    assert res1[0]["object"] == "B"
    
    res2 = graph.query_point_in_time(5.0)
    assert len(res2) == 1
    assert res2[0]["object"] == "C"
    
    res3 = graph.query_point_in_time(10.0)
    assert len(res3) == 0

def test_query_entity_history():
    graph = TemporalEntityGraph()
    graph.add_relation("A", "knows", "B", valid_from=10.0)
    graph.add_relation("C", "knows", "A", valid_from=5.0)
    
    hist = graph.query_entity_history("A")
    assert len(hist) == 2
    assert hist[0]["subject"] == "C"
    assert hist[1]["subject"] == "A"

@pytest.mark.xfail(reason="Mermaid sanitizer collapses IDs differing only by stripped characters")
def test_export_graph_mermaid_sanitizer_collision():
    """
    Test that registers two distinct entities whose IDs differ only by characters 
    stripped by the Mermaid sanitizer (for example "a-b" and "a_b", or "node.1" and "node/1") 
    and asserts that export_graph_mermaid emits two separate, uniquely identifiable node declarations 
    rather than collapsing.
    """
    graph = TemporalEntityGraph()
    graph.add_entity("node.1")
    graph.add_entity("node/1")
    
    mermaid_out = graph.export_graph_mermaid()
    
    lines = [line.strip() for line in mermaid_out.split('\n')]
    
    node_1_lines = [line for line in lines if '["node.1"]' in line]
    node_slash_1_lines = [line for line in lines if '["node/1"]' in line]
    
    assert len(node_1_lines) == 1, "Should emit declaration for node.1"
    assert len(node_slash_1_lines) == 1, "Should emit declaration for node/1"
    
    id1 = node_1_lines[0].split('[')[0]
    id2 = node_slash_1_lines[0].split('[')[0]
    
    assert id1 != id2, "Mermaid IDs for node.1 and node/1 must be uniquely identifiable and not collapse"

def test_extract_temporal_knowledge_graph():
    steps = [
        {"timestamp": 1.0, "sender": "Agent1", "recipient": "Agent2", "type": "message", "step_id": "s1"},
        {"timestamp": 2.0, "sender": "Agent1", "target_file": "file.txt", "type": "write"},
        {"timestamp": 3.0, "sender": "Agent2", "test_target": "test_1", "type": "test", "outcome": "pass"}
    ]
    graph = extract_temporal_knowledge_graph(steps)
    
    assert "Agent1" in graph.entities
    assert "Agent2" in graph.entities
    assert "file.txt" in graph.entities
    assert "test_1" in graph.entities
    
    rels = graph.relations
    assert len(rels) == 3
    
    assert rels[0]["subject"] == "Agent1"
    assert rels[0]["object"] == "Agent2"
    
    assert rels[1]["subject"] == "Agent1"
    assert rels[1]["object"] == "file.txt"
    
    assert rels[2]["subject"] == "Agent2"
    assert rels[2]["object"] == "test_1"
    assert rels[2]["provenance"]["outcome"] == "pass"

def test_extract_graph_with_window():
    steps = [
        {"timestamp": 1.0, "sender": "A", "recipient": "B"},
        {"timestamp": 5.0, "sender": "B", "recipient": "C"}
    ]
    graph = extract_temporal_knowledge_graph(steps, time_window=(2.0, 10.0))
    
    assert "A" not in graph.entities
    assert "B" in graph.entities
    assert "C" in graph.entities
    assert len(graph.relations) == 1
    assert graph.relations[0]["object"] == "C"

def test_export_graph_json():
    graph = TemporalEntityGraph()
    graph.add_entity("E1")
    graph.add_relation("E1", "rel", "E2")
    
    data = graph.export_graph_json()
    assert data["entity_count"] == 2
    assert data["relation_count"] == 1
    assert "entities" in data
    assert "relations" in data

def test_query_point_in_time_edge_cases():
    graph = TemporalEntityGraph()
    graph.add_relation("A", "p", "B", valid_from=10.0, valid_to=20.0)
    
    assert len(graph.query_point_in_time(9.9)) == 0
    assert len(graph.query_point_in_time(10.0)) == 1
    assert len(graph.query_point_in_time(15.0)) == 1
    assert len(graph.query_point_in_time(19.9)) == 1
    assert len(graph.query_point_in_time(20.0)) == 0
    
    assert len(graph.query_point_in_time(15.0, entity_id="A")) == 1
    assert len(graph.query_point_in_time(15.0, entity_id="C")) == 0
    
    assert len(graph.query_point_in_time(15.0, predicate="p")) == 1
    assert len(graph.query_point_in_time(15.0, predicate="q")) == 0

def test_extract_temporal_knowledge_graph_empty_and_invalid():
    assert len(extract_temporal_knowledge_graph(None).entities) == 0
    assert len(extract_temporal_knowledge_graph([]).entities) == 0
    
    steps = [{"not_a_dict": True}, "string_step", {"timestamp": 1.0}]
    graph = extract_temporal_knowledge_graph(steps)
    assert len(graph.entities) == 0

def test_extract_graph_sender_only():
    steps = [
        {"timestamp": 1.0, "sender": "AgentOnly", "step_id": "123"}
    ]
    graph = extract_temporal_knowledge_graph(steps)
    assert "AgentOnly" in graph.entities
    assert len(graph.relations) == 1
    assert graph.relations[0]["subject"] == "AgentOnly"
    assert graph.relations[0]["object"] == "step_123"

def test_graph_mermaid_direction():
    graph = TemporalEntityGraph()
    out = graph.export_graph_mermaid("TD")
    assert out.startswith("flowchart TD")

def test_add_entity_updates_existing():
    graph = TemporalEntityGraph()
    graph.add_entity("E1", "Type1", {"a": 1}, 1.0)
    graph.add_entity("E1", "Type2", {"b": 2}, 2.0)
    
    assert len(graph.entities) == 1
    assert graph.entities["E1"]["entity_type"] == "Type2"
    assert graph.entities["E1"]["properties"] == {"b": 2}
    assert graph.entities["E1"]["timestamp"] == 2.0
