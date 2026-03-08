"""
Définition du graphe LangGraph
"""
from langgraph.graph import StateGraph, END
from langgraph_agent.state import AgentState
from langgraph_agent.nodes import WorkflowNodes

def create_workflow(groq_api_key: str) -> StateGraph:
    """Crée le workflow LangGraph"""
    
    # Initialiser les nœuds
    nodes = WorkflowNodes(groq_api_key)
    
    # Créer le graphe
    workflow = StateGraph(AgentState)
    
    # Ajouter les nœuds
    workflow.add_node("analyze_failure", nodes.analyze_failure)
    workflow.add_node("identify_cause", nodes.identify_cause)
    workflow.add_node("query_rag", nodes.query_rag)
    workflow.add_node("generate_fix", nodes.generate_fix)
    workflow.add_node("validate_fix", nodes.validate_fix)
    workflow.add_node("calculate_confidence", nodes.calculate_confidence)
    
    # Définir les edges (flux)
    workflow.set_entry_point("analyze_failure")
    workflow.add_edge("analyze_failure", "identify_cause")
    workflow.add_edge("identify_cause", "query_rag")
    workflow.add_edge("query_rag", "generate_fix")
    workflow.add_edge("generate_fix", "validate_fix")
    workflow.add_edge("validate_fix", "calculate_confidence")
    workflow.add_edge("calculate_confidence", END)
    
    return workflow.compile()