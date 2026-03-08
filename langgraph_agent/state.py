from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum

class ErrorType(str, Enum):
    """Types d'erreurs détectées"""
    MOCK_MISSING = "mock_missing"
    KOIN_MISSING = "koin_missing"
    ASSERTION_ERROR = "assertion_error"
    COROUTINE_DISPATCHER = "coroutine_dispatcher"
    NULL_POINTER = "null_pointer"
    UNKNOWN = "unknown"

class AgentState(TypedDict):
    """État global du workflow"""
    # Inputs
    test_file: str
    test_name: str
    test_code: str
    error_logs: str
    
    # Analyse
    error_type: Optional[ErrorType]
    error_message: Optional[str]
    stack_trace: Optional[List[str]]
    
    # Contexte RAG
    similar_tests: Optional[List[Dict[str, Any]]]
    similar_fixes: Optional[List[Dict[str, Any]]]
    project_conventions: Optional[Dict[str, Any]]
    
    # Correction
    proposed_fix: Optional[str]
    explanation: Optional[str]
    confidence_score: Optional[float]
    
    # Validation
    is_valid_kotlin: Optional[bool]
    validation_errors: Optional[List[str]]
    
    # Metadata
    tokens_used: int
    processing_time: float
    steps_completed: List[str]