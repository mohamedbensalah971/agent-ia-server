"""
Retriever - RAG System
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from rag_system.chromadb_client import get_chromadb_client


class RAGRetriever:
    def __init__(self):
        self.chroma_client = get_chromadb_client()
    
    def get_context_for_fix(self, test_code: str, error_logs: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"🔍 Retrieving RAG context for fix (error_type={error_type})")

        # Build meaningful queries from actual inputs instead of hardcoded strings
        test_query = test_code[:500]       # class/function names + annotations
        error_query = error_logs[:300]     # most relevant part of the error

        similar_tests = self.chroma_client.search_similar_tests(query=test_query, n_results=3)
        similar_fixes = self.chroma_client.search_fixes(query=error_query, error_type=error_type, n_results=3)
        conventions = self.chroma_client.search_conventions(query=error_query, n_results=4)

        logger.info(f"   similar_tests={len(similar_tests)}, fixes={len(similar_fixes)}, conventions={len(conventions)}")

        return {
            "similar_tests": similar_tests,
            "similar_fixes": similar_fixes,
            "conventions": conventions,
            "error_type": error_type
        }
    
    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        formatted = []
        
        if context.get("conventions"):
            formatted.append("=== PROJECT CONVENTIONS ===")
            for conv in context["conventions"][:3]:
                formatted.append(f"- {conv['description']}")
        
        if context.get("similar_tests"):
            formatted.append("\n=== SIMILAR TESTS FROM PROJECT ===")
            for test in context["similar_tests"][:2]:
                formatted.append(test["code"][:300])

        if context.get("similar_fixes"):
            formatted.append("\n=== KNOWN FIXES FOR THIS ERROR TYPE ===")
            for fix in context["similar_fixes"][:2]:
                formatted.append(fix["fix_code"][:300])
        
        return "\n".join(formatted)


_rag_retriever = None

def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever
