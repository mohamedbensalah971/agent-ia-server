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

        # Convert conventions list to a readable dict for the generate_fix prompt.
        # Some retrieval paths return `content` only (without `description`).
        project_conventions = {}
        for conv in context.get("conventions", []):
            if not isinstance(conv, dict):
                continue

            category = conv.get("category") or "general"
            description = (
                conv.get("description")
                or conv.get("content")
                or (conv.get("metadata") or {}).get("description")
                or ""
            )

            if description:
                # Keep the richest description if the same category appears multiple times.
                existing = project_conventions.get(category, "")
                if len(description) > len(existing):
                    project_conventions[category] = description

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

        prompt = f"""═══════════════════════════════════════════════════════════════════
TASK: Fix the failing Kotlin Android test
═══════════════════════════════════════════════════════════════════

FAILING TEST CODE:
```kotlin
{state['test_code']}
```

ERROR MESSAGE:
{state['error_message']}

ERROR TYPE: {state['error_type']}

═══════════════════════════════════════════════════════════════════
PROJECT CONTEXT (from knowledge base)
═══════════════════════════════════════════════════════════════════
{rag_context if rag_context else "No additional context available"}

═══════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════
1. Return ONLY the corrected Kotlin code in a ```kotlin``` block
2. Spell all annotations exactly: @Test, @BeforeEach, @AfterEach
3. Use JUnit 5 imports (org.junit.jupiter.api.*)
4. Use MockK syntax correctly: every {{ }}, verify {{ }}
5. Balance all braces and parentheses
6. Do NOT include explanations outside the code block

CORRECT ANNOTATION EXAMPLES:
- @Test (NOT @Testt)
- @BeforeEach (NOT @BeforeEachEach)
- assertEquals(expected, actual)
- every {{ mock.method() }} returns value
"""
        
        # Appel Groq
        response = self.llm.invoke(prompt)
        proposed_fix = response.content
        
        # Générer explication
        explanation_prompt = f"""Explain in 2-3 short sentences why this fix resolves the issue:

ORIGINAL CODE:
{state['test_code'][:500]}

FIXED CODE:
{proposed_fix[:500]}

ERROR:
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
        """Nœud 5: Valider la syntaxe Kotlin avec détection de typos"""
        print("✅ Validation de la syntaxe...")
        
        proposed_fix = state.get("proposed_fix", "")
        
        validation_errors = []
        is_valid = True
        
        if not proposed_fix or not proposed_fix.strip():
            validation_errors.append("Code vide généré")
            return {
                "is_valid_kotlin": False,
                "validation_errors": validation_errors,
                "steps_completed": state["steps_completed"] + ["validate_fix"]
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # ANNOTATION TYPO DETECTION
        # ═══════════════════════════════════════════════════════════════════
        
        # Common @Test typos
        test_typos = re.findall(r'@[Tt][Ee][Ss][Tt]+(?![a-zA-Z])', proposed_fix)
        for typo in test_typos:
            if typo != "@Test":
                validation_errors.append(f"Typo annotation: '{typo}' → '@Test'")
                is_valid = False
        
        # Common @BeforeEach typos
        before_typos = re.findall(r'@[Bb]efore[Ee]ach[Ee]?a?c?h?', proposed_fix)
        for typo in before_typos:
            if typo != "@BeforeEach":
                validation_errors.append(f"Typo annotation: '{typo}' → '@BeforeEach'")
                is_valid = False
        
        # Common @AfterEach typos
        after_typos = re.findall(r'@[Aa]fter[Ee]ach[Ee]?a?c?h?', proposed_fix)
        for typo in after_typos:
            if typo != "@AfterEach":
                validation_errors.append(f"Typo annotation: '{typo}' → '@AfterEach'")
                is_valid = False
        
        # Detect doubled annotations (@TestTest, @BeforeEachEach)
        doubled = re.findall(r'@(\w+)\1', proposed_fix)
        for match in doubled:
            validation_errors.append(f"Annotation doublée: '@{match}{match}'")
            is_valid = False
        
        # ═══════════════════════════════════════════════════════════════════
        # STRUCTURE VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
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
            validation_errors.append(f"Accolades non équilibrées: {proposed_fix.count('{')} '{{' vs {proposed_fix.count('}')} '}}'")
            is_valid = False
        
        # Check 4: Parenthèses équilibrées
        if proposed_fix.count("(") != proposed_fix.count(")"):
            validation_errors.append(f"Parenthèses non équilibrées: {proposed_fix.count('(')} '(' vs {proposed_fix.count(')')} ')'")
            is_valid = False
        
        # ═══════════════════════════════════════════════════════════════════
        # JUNIT IMPORT VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
        if "org.junit.Test" in proposed_fix and "org.junit.jupiter" not in proposed_fix:
            validation_errors.append("Import JUnit 4 détecté: utiliser org.junit.jupiter.api.Test")
            is_valid = False
        
        if "@RunWith" in proposed_fix:
            validation_errors.append("@RunWith (JUnit 4) détecté: non compatible avec JUnit 5")
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