import re
import time
from typing import Dict, Any
from langchain_groq import ChatGroq
from langgraph_agent.state import AgentState, ErrorType
from rag_system.retriever import get_rag_retriever

class WorkflowNodes:
    """Tous les nœuds du workflow"""
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_api_key
        )
        self.rag_retriever = get_rag_retriever()
    
    def analyze_failure(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 1: Analyser l'échec du test"""
        print("Analyse de l'échec")
        
        error_logs = state["error_logs"]
        
        # Identifier le type d'erreur
        error_type = ErrorType.UNKNOWN
        
        if "NullPointerException" in error_logs:
            error_type = ErrorType.NULL_POINTER
        elif "MockK" in error_logs or "mock" in error_logs.lower():
            error_type = ErrorType.MOCK_MISSING
        elif "Koin" in error_logs or "inject" in error_logs:
            error_type = ErrorType.KOIN_MISSING
        elif "AssertionError" in error_logs or "expected" in error_logs.lower():
            error_type = ErrorType.ASSERTION_ERROR
        elif "Dispatcher" in error_logs or "coroutine" in error_logs.lower():
            error_type = ErrorType.COROUTINE_DISPATCHER
        
        # Extraire stack trace
        stack_lines = [line.strip() for line in error_logs.split('\n') if line.strip().startswith('at ')]
        
        return {
            "error_type": error_type,
            "error_message": error_logs.split('\n')[0] if error_logs else "",
            "stack_trace": stack_lines,
            "steps_completed": state.get("steps_completed", []) + ["analyze_failure"]
        }
    
    def identify_cause(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 2: Identifier la cause racine"""
        print("🎯 Identification de la cause racine...")
        
        error_type = state["error_type"]
        test_code = state["test_code"]
        
        # Analyse contextuelle basée sur le type d'erreur
        cause_analysis = {
            ErrorType.MOCK_MISSING: "Mock non initialisé ou absent",
            ErrorType.KOIN_MISSING: "Dépendance Koin non injectée",
            ErrorType.NULL_POINTER: "Variable null non initialisée",
            ErrorType.ASSERTION_ERROR: "Valeur attendue différente de la valeur réelle",
            ErrorType.COROUTINE_DISPATCHER: "Dispatcher coroutine mal configuré"
        }.get(error_type, "Cause inconnue")
        
        return {
            "error_message": f"{state['error_message']} - {cause_analysis}",
            "steps_completed": state["steps_completed"] + ["identify_cause"]
        }
    
    def query_rag(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 3: Requête RAG - interroge ChromaDB pour du contexte réel"""
        print("📚 Query RAG (ChromaDB)...")

        context = self.rag_retriever.get_context_for_fix(
            test_code=state["test_code"],
            error_logs=state["error_logs"],
            error_type=state["error_type"].value if state.get("error_type") else None
        )

        # Convert conventions list to a readable dict for the generate_fix prompt
        project_conventions = {
            conv.get("category", "general"): conv["description"]
            for conv in context["conventions"]
        }

        return {
            "similar_tests": context["similar_tests"],
            "similar_fixes": context["similar_fixes"],
            "project_conventions": project_conventions,
            "steps_completed": state["steps_completed"] + ["query_rag"]
        }
    
    def generate_fix(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 4: Générer la correction via Groq"""
        print("🤖 Génération de la correction via Groq...")
        
        start_time = time.time()
        
        # Construire le prompt
        rag_context = self.rag_retriever.format_context_for_prompt({
            "conventions": [
                {"description": f"{cat}: {desc}"}
                for cat, desc in state.get("project_conventions", {}).items()
            ],
            "similar_tests": state.get("similar_tests", []),
            "similar_fixes": state.get("similar_fixes", []),
        })

        prompt = f"""Tu es un expert en tests Kotlin Android.

TEST ÉCHOUÉ:
{state['test_code']}

ERREUR:
{state['error_message']}

TYPE D'ERREUR: {state['error_type']}

CONTEXTE DU PROJET (issu de la base de connaissances):
{rag_context if rag_context else "Aucun contexte disponible"}

TÂCHE:
Génère UNIQUEMENT le code corrigé complet du test.
Ne donne PAS d'explication, juste le code Kotlin.
Respecte les conventions du projet.
"""
        
        # Appel Groq
        response = self.llm.invoke(prompt)
        proposed_fix = response.content
        
        # Générer explication
        explanation_prompt = f"""Explique en 2-3 phrases courtes pourquoi cette correction résout le problème:

CODE ORIGINAL:
{state['test_code']}

CODE CORRIGÉ:
{proposed_fix}

ERREUR:
{state['error_message']}
"""
        
        explanation_response = self.llm.invoke(explanation_prompt)
        explanation = explanation_response.content
        
        processing_time = time.time() - start_time
        
        return {
            "proposed_fix": proposed_fix,
            "explanation": explanation,
            "processing_time": processing_time,
            "tokens_used": state.get("tokens_used", 0) + 500,  # Estimation
            "steps_completed": state["steps_completed"] + ["generate_fix"]
        }
    
    def validate_fix(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 5: Valider la syntaxe Kotlin"""
        print("✅ Validation de la syntaxe...")
        
        proposed_fix = state.get("proposed_fix", "")
        
        # Validations basiques
        validation_errors = []
        is_valid = True
        
        # Check 1: Le code contient @Test
        if "@Test" not in proposed_fix:
            validation_errors.append("Manque annotation @Test")
            is_valid = False
        
        # Check 2: Le code contient fun
        if "fun " not in proposed_fix:
            validation_errors.append("Manque déclaration de fonction")
            is_valid = False
        
        # Check 3: Accolades équilibrées
        if proposed_fix.count("{") != proposed_fix.count("}"):
            validation_errors.append("Accolades non équilibrées")
            is_valid = False
        
        return {
            "is_valid_kotlin": is_valid,
            "validation_errors": validation_errors,
            "steps_completed": state["steps_completed"] + ["validate_fix"]
        }
    
    def calculate_confidence(self, state: AgentState) -> Dict[str, Any]:
        """Nœud 6: Calculer le score de confiance"""
        print("🎲 Calcul du score de confiance...")
        
        confidence = 0.5  # Base
        
        # Facteurs augmentant la confiance
        if state.get("is_valid_kotlin"):
            confidence += 0.2
        
        if state.get("error_type") in [ErrorType.MOCK_MISSING, ErrorType.KOIN_MISSING]:
            confidence += 0.15  # Erreurs bien connues
        
        if state.get("similar_tests"):
            confidence += 0.1
        
        if not state.get("validation_errors"):
            confidence += 0.05
        
        # Limiter entre 0 et 1
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            "confidence_score": confidence,
            "steps_completed": state["steps_completed"] + ["calculate_confidence"]
        }