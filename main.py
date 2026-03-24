"""
AI Agent Server - Main FastAPI Application
Supports both direct Groq calls and LangGraph workflow
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import re
import uvicorn
from loguru import logger
import sys

from config import settings, validate_settings
from groq_client import get_groq_client

# LangGraph imports
try:
    from langgraph_agent.graph import create_workflow
    from langgraph_agent.state import AgentState
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph modules loaded successfully")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logger.warning(f"⚠️ LangGraph not available: {e}")
    logger.warning("   Install with: pip install langgraph langchain langchain-groq")

# Configure logging
logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL)
logger.add(
    settings.LOG_FILE,
    rotation="10 MB",
    retention="7 days",
    level=settings.LOG_LEVEL
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered test automation and correction system with LangGraph support"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# DATA MODELS
# ==========================================

class TestFailureRequest(BaseModel):
    """Request model for test failure analysis"""
    test_file: str = Field(..., description="Path to the test file")
    test_name: str = Field(..., description="Name of the failing test")
    test_code: str = Field(..., description="The test code that failed")
    error_logs: str = Field(..., description="Error logs from test execution")
    source_code: Optional[str] = Field(None, description="Source code being tested (optional)")
    jenkins_build_url: Optional[str] = Field(None, description="Jenkins build URL (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_file": "app/src/test/java/com/smarttalk/UserManagerTest.kt",
                "test_name": "testUserLogin",
                "test_code": "@Test\nfun testUserLogin() {\n    val user = userManager.login(\"test@example.com\", \"password\")\n    assertEquals(\"test@example.com\", user.email)\n}",
                "error_logs": "java.lang.NullPointerException: userManager is null\n    at com.smarttalk.UserManagerTest.testUserLogin(UserManagerTest.kt:25)",
                "source_code": "class UserManager { ... }",
                "jenkins_build_url": "http://jenkins.local/job/SmartTalk/123"
            }
        }


class CorrectionResponse(BaseModel):
    """Response model for correction"""
    success: bool
    correction_id: str
    corrected_code: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    rate_limit_exceeded: Optional[bool] = False
    timestamp: datetime = Field(default_factory=datetime.now)


class LangGraphCorrectionResponse(BaseModel):
    """Response model for LangGraph workflow correction"""
    success: bool
    correction_id: str
    test_file: str
    test_name: str
    error_type: Optional[str] = None
    proposed_fix: Optional[str] = None
    explanation: Optional[str] = None
    confidence_score: Optional[float] = None
    is_valid: Optional[bool] = None
    validation_errors: Optional[List[str]] = None
    tokens_used: int
    processing_time: float
    steps_completed: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class ApprovalRequest(BaseModel):
    """Request model for correction approval"""
    correction_id: str
    approved: bool
    feedback: Optional[str] = None
    # Phase 7: context fields needed to persist the fix into RAG.
    test_code: Optional[str] = Field(None, description="Original failing test code")
    fix_code: Optional[str] = Field(None, description="The corrected test code")
    error_type: Optional[str] = Field(None, description="Detected error type")
    error_message: Optional[str] = Field(None, description="Error message")
    test_file: Optional[str] = Field(None, description="Path of the test file")
    confidence_score: Optional[float] = Field(None, description="Model confidence score")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    groq_api: str
    langgraph_available: bool
    tokens_available: Dict[str, Any]


class TestGenerationRequest(BaseModel):
    """Request model for generating new unit tests from source code."""
    source_file: str = Field(..., description="Path to source file under test")
    source_code: str = Field(..., description="Source Kotlin code to generate tests for")
    class_name: Optional[str] = Field(None, description="Target class name (optional)")
    existing_tests: Optional[str] = Field(None, description="Existing tests to avoid duplicates")
    framework: str = Field("junit5_mockk", description="Test framework profile")
    include_edge_cases: bool = Field(True, description="Generate edge-case tests")
    max_tests: int = Field(6, ge=1, le=20, description="Approximate max number of tests")
    use_rag: bool = Field(True, description="Use RAG context from project knowledge base")
    strict_mode: bool = Field(True, description="Apply post-generation cleanup and fail if risky patterns remain")


class TestGenerationResponse(BaseModel):
    """Response model for generated unit tests."""
    success: bool
    generation_id: str
    generated_tests: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    rag_context_used: bool = False
    quality_notes: Optional[List[str]] = None
    error: Optional[str] = None
    rate_limit_exceeded: Optional[bool] = False
    timestamp: datetime = Field(default_factory=datetime.now)


def _post_process_generated_tests(code: str) -> Tuple[str, List[str], List[str]]:
    """Normalize generated tests to safer project conventions and detect risky leftovers."""
    notes: List[str] = []
    unresolved_issues: List[str] = []

    if not code.strip():
        return code, notes, ["Empty generated test output"]

    lines = code.splitlines()
    cleaned_lines: List[str] = []

    banned_import_contains = [
        "org.junit.runner.RunWith",
        "org.junit.runners.JUnit4",
        "org.junit.Rule",
        "kotlinx.coroutines.test.runTest",
        "kotlinx.coroutines.ExperimentalCoroutinesApi",
        "BaseBindingModel",
    ]

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and any(token in stripped for token in banned_import_contains):
            notes.append(f"Removed risky import: {stripped}")
            continue

        if stripped == "import org.junit.Test":
            notes.append("Replaced JUnit4 Test import with JUnit5")
            cleaned_lines.append(line.replace("import org.junit.Test", "import org.junit.jupiter.api.Test"))
            continue

        if stripped == "import org.junit.Before":
            notes.append("Replaced JUnit4 Before import with JUnit5")
            cleaned_lines.append(line.replace("import org.junit.Before", "import org.junit.jupiter.api.BeforeEach"))
            continue

        if stripped.startswith("@RunWith(") or stripped == "@ExperimentalCoroutinesApi":
            notes.append(f"Removed annotation: {stripped}")
            continue

        if "@get:Rule" in line or "TestCoroutineRule(" in line:
            notes.append("Removed TestCoroutineRule usage")
            continue

        updated_line = line
        updated_line = updated_line.replace("@Before", "@BeforeEach")
        updated_line = updated_line.replace(" = runTest {", " {")
        cleaned_lines.append(updated_line)

    cleaned_code = "\n".join(cleaned_lines)

    # Normalize Kotlin assert(...) calls to JUnit5 assertions where possible.
    def _replace_not(m):
        expr = m.group(1).strip()
        return f"assertFalse({expr})"

    def _replace_pos(m):
        expr = m.group(1).strip()
        return f"assertTrue({expr})"

    new_code = re.sub(r"assert\s*\(\s*!\s*(.*?)\s*\)", _replace_not, cleaned_code)
    new_code = re.sub(r"assert\s*\(([^\)]*?)\)", _replace_pos, new_code)
    if new_code != cleaned_code:
        notes.append("Normalized Kotlin assert(...) to JUnit-style assertTrue/assertFalse")
    cleaned_code = new_code

    # Ensure JUnit5 assertions import exists if assertions are used.
    if ("assertTrue(" in cleaned_code or "assertFalse(" in cleaned_code) and "import org.junit.jupiter.api.Assertions" not in cleaned_code:
        cleaned_code = "import org.junit.jupiter.api.Assertions.*\n" + cleaned_code
        notes.append("Added JUnit5 Assertions import")

    # Detect unresolved risky patterns.
    risk_patterns = {
        "@RunWith": "JUnit4 RunWith annotation still present",
        "JUnit4": "JUnit4 class reference still present",
        "@get:Rule": "Rule annotation still present",
        "TestCoroutineRule": "TestCoroutineRule still present",
        " = runTest {": "runTest still present",
        "import org.junit.Test": "JUnit4 Test import still present",
        "import org.junit.Before": "JUnit4 Before import still present",
    }
    for token, message in risk_patterns.items():
        if token in cleaned_code:
            unresolved_issues.append(message)

    return cleaned_code, notes, unresolved_issues


# ==========================================
# STARTUP & SHUTDOWN
# ==========================================

# In-memory store for pending corrections (correction_id -> context).
# Used by /approve-correction to persist fixes into RAG without a separate DB.
_pending_corrections: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)
    
    try:
        validate_settings()
        
        # Initialize Groq client
        groq_client = get_groq_client()
        logger.info("✅ Groq client initialized")
        
        # Check LangGraph availability
        if LANGGRAPH_AVAILABLE:
            logger.info("✅ LangGraph workflow available")
            from rag_system.retriever import get_rag_retriever

            logger.info("🔥 Warming LangGraph + RAG components...")
            create_workflow(settings.GROQ_API_KEY)
            get_rag_retriever().warmup()
        else:
            logger.warning("⚠️ LangGraph not installed - only basic endpoint available")
        
        # Log configuration
        logger.info(f"Model: {settings.GROQ_MODEL}")
        logger.info(f"Git Repo: {settings.GIT_REPO_PATH}")
        logger.info(f"Cache: {'Enabled' if settings.CACHE_ENABLED else 'Disabled'}")
        logger.info(f"Rate Limits: {settings.RATE_LIMIT_TOKENS_PER_MINUTE}/min, {settings.RATE_LIMIT_TOKENS_PER_DAY}/day")
        
        logger.info("=" * 60)
        logger.info("✅ Server ready!")
        logger.info(f"📍 Listening on http://{settings.HOST}:{settings.PORT}")
        if settings.HOST in ("0.0.0.0", "::"):
            logger.info(f"🌐 Open in browser: http://127.0.0.1:{settings.PORT}")
            logger.info(f"📚 API Docs: http://127.0.0.1:{settings.PORT}/docs")
            logger.info(f"📚 API Docs (alt): http://localhost:{settings.PORT}/docs")
        else:
            logger.info(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Shutting down server...")


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "basic": "/analyze-failure",
            "langgraph": "/analyze-failure-langgraph" if LANGGRAPH_AVAILABLE else "Not available",
            "generate_tests": "/generate-tests"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns server status and resource availability
    """
    groq_client = get_groq_client()
    stats = groq_client.get_stats()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        groq_api="connected",
        langgraph_available=LANGGRAPH_AVAILABLE,
        tokens_available={
            "tokens_used_today": stats["tokens_used_day"],
            "tokens_remaining_today": stats["rate_limit_day"] - stats["tokens_used_day"],
            "percentage_used": round((stats["tokens_used_day"] / stats["rate_limit_day"]) * 100, 2)
        }
    )


@app.post("/analyze-failure", response_model=CorrectionResponse, tags=["Corrections"])
async def analyze_test_failure(request: TestFailureRequest):
    """
    Analyze a failing test and generate correction (Basic endpoint)
    
    This is the simple endpoint using direct Groq calls (Phase 1)
    For advanced workflow, use /analyze-failure-langgraph
    """
    logger.info(f"📥 [BASIC] Received analysis request for test: {request.test_name}")
    logger.debug(f"Test file: {request.test_file}")
    
    try:
        groq_client = get_groq_client()
        
        # Generate correction
        result = groq_client.generate_correction(
            test_code=request.test_code,
            error_logs=request.error_logs,
            source_code=request.source_code,
            context=None  # RAG context will be added in Phase 3
        )
        
        if not result["success"]:
            logger.error(f"❌ Correction generation failed: {result.get('error')}")
            return CorrectionResponse(
                success=False,
                correction_id="",
                error=result.get("error"),
                rate_limit_exceeded=result.get("rate_limit_exceeded", False)
            )
        
        # Generate unique correction ID
        correction_id = f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Phase 7: cache correction context for feedback loop.
        _pending_corrections[correction_id] = {
            "test_code": request.test_code,
            "fix_code": result.get("corrected_code", ""),
            "error_type": None,
            "error_message": request.error_logs[:300],
            "test_file": request.test_file,
            "confidence_score": result.get("confidence", 0.0),
        }

        logger.info(f"✅ Correction generated: {correction_id}")
        logger.info(f"   Confidence: {result['confidence']}")
        logger.info(f"   Tokens used: {result['tokens_used']}")
        
        return CorrectionResponse(
            success=True,
            correction_id=correction_id,
            corrected_code=result["corrected_code"],
            explanation=result["explanation"],
            confidence=result["confidence"],
            tokens_used=result["tokens_used"]
        )
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-tests", response_model=TestGenerationResponse, tags=["Generation"])
async def generate_tests(request: TestGenerationRequest):
    """
    Generate new Kotlin unit tests from source code.

    This endpoint is for test creation (not failure correction).
    It can optionally use RAG to align generated tests with project conventions and patterns.
    """
    logger.info(f"🧪 [GEN] Received test generation request for: {request.source_file}")

    try:
        groq_client = get_groq_client()

        rag_context_text = None
        rag_context_used = False
        if request.use_rag:
            try:
                from rag_system.retriever import get_rag_retriever

                retriever = get_rag_retriever()
                context = retriever.get_context_for_fix(
                    test_code=request.source_code,
                    error_logs=f"generate unit tests for {request.class_name or request.source_file}",
                    error_type="test_generation",
                )
                rag_context_text = retriever.format_context_for_prompt(context)
                rag_context_used = bool((rag_context_text or "").strip())
            except Exception as e:
                logger.warning(f"⚠️ RAG context unavailable for test generation, continuing without it: {e}")

        result = groq_client.generate_unit_tests(
            source_code=request.source_code,
            class_name=request.class_name,
            existing_tests=request.existing_tests,
            framework=request.framework,
            include_edge_cases=request.include_edge_cases,
            max_tests=request.max_tests,
            rag_context=rag_context_text,
        )

        if not result.get("success"):
            return TestGenerationResponse(
                success=False,
                generation_id="",
                error=result.get("error"),
                rate_limit_exceeded=result.get("rate_limit_exceeded", False),
                rag_context_used=rag_context_used,
            )

        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generated_tests = result.get("generated_tests") or ""
        cleaned_tests, quality_notes, unresolved_issues = _post_process_generated_tests(generated_tests)

        if request.strict_mode and unresolved_issues:
            logger.warning(f"⚠️ Strict mode rejected generated tests: {unresolved_issues}")
            return TestGenerationResponse(
                success=False,
                generation_id=generation_id,
                generated_tests=cleaned_tests,
                explanation=result.get("explanation"),
                confidence=result.get("confidence"),
                tokens_used=result.get("tokens_used"),
                rag_context_used=rag_context_used,
                quality_notes=quality_notes + unresolved_issues,
                error="Generated tests contain risky patterns in strict mode. Review quality_notes.",
            )

        logger.info(f"✅ Unit tests generated: {generation_id}")

        return TestGenerationResponse(
            success=True,
            generation_id=generation_id,
            generated_tests=cleaned_tests,
            explanation=result.get("explanation"),
            confidence=result.get("confidence"),
            tokens_used=result.get("tokens_used"),
            rag_context_used=rag_context_used,
            quality_notes=quality_notes,
        )
    except Exception as e:
        logger.error(f"❌ Test generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")


@app.post("/analyze-failure-langgraph", response_model=LangGraphCorrectionResponse, tags=["Corrections"])
async def analyze_failure_langgraph(request: TestFailureRequest):
    """
    Analyze a failing test using LangGraph workflow (Advanced endpoint - Phase 2)
    
    This endpoint uses a sophisticated state machine workflow with:
    - Step-by-step analysis
    - RAG context integration (Phase 3)
    - Validation and confidence scoring
    - Full traceability
    
    Requires: LangGraph, LangChain, langchain-groq
    Install: pip install langgraph langchain langchain-groq
    """
    if not LANGGRAPH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LangGraph not available. Install with: pip install langgraph langchain langchain-groq"
        )
    
    logger.info(f"📥 [LANGGRAPH] Received analysis request for test: {request.test_name}")
    logger.debug(f"Test file: {request.test_file}")
    
    try:
        # Create the workflow
        logger.info("🔄 Getting LangGraph workflow...")
        workflow = create_workflow(settings.GROQ_API_KEY)
        
        # Prepare initial state
        initial_state: AgentState = {
            "test_file": request.test_file,
            "test_name": request.test_name,
            "test_code": request.test_code,
            "error_logs": request.error_logs,
            "error_type": None,
            "error_message": None,
            "stack_trace": None,
            "similar_tests": None,
            "similar_fixes": None,
            "project_conventions": None,
            "proposed_fix": None,
            "explanation": None,
            "confidence_score": None,
            "is_valid_kotlin": None,
            "validation_errors": None,
            "tokens_used": 0,
            "processing_time": 0.0,
            "steps_completed": []
        }
        
        # Execute the workflow
        logger.info("▶️ Executing LangGraph workflow...")
        result = workflow.invoke(initial_state)
        
        # Generate unique correction ID
        correction_id = f"lg_corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Phase 7: cache correction context for feedback loop.
        _pending_corrections[correction_id] = {
            "test_code": request.test_code,
            "fix_code": result.get("proposed_fix", ""),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message", request.error_logs[:300]),
            "test_file": request.test_file,
            "confidence_score": result.get("confidence_score", 0.0),
        }

        logger.info(f"✅ LangGraph workflow completed: {correction_id}")
        logger.info(f"   Steps completed: {len(result['steps_completed'])}")
        logger.info(f"   Error type: {result['error_type']}")
        logger.info(f"   Confidence: {result['confidence_score']}")
        logger.info(f"   Valid Kotlin: {result['is_valid_kotlin']}")
        logger.info(f"   Processing time: {result['processing_time']:.2f}s")
        logger.info(f"   Tokens used: {result['tokens_used']}")
        
        # Return the result
        return LangGraphCorrectionResponse(
            success=True,
            correction_id=correction_id,
            test_file=result["test_file"],
            test_name=result["test_name"],
            error_type=result["error_type"],
            proposed_fix=result["proposed_fix"],
            explanation=result["explanation"],
            confidence_score=result["confidence_score"],
            is_valid=result["is_valid_kotlin"],
            validation_errors=result.get("validation_errors", []),
            tokens_used=result["tokens_used"],
            processing_time=result["processing_time"],
            steps_completed=result["steps_completed"]
        )
        
    except Exception as e:
        logger.error(f"❌ LangGraph workflow error: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=f"LangGraph workflow failed: {str(e)}")


@app.post("/approve-correction", tags=["Corrections"])
async def approve_correction(request: ApprovalRequest, background_tasks: BackgroundTasks):
    """
    Approve or reject a correction.

    When approved, the fix is stored into the RAG vector database (test_fixes collection)
    so future similar errors benefit from this human-validated solution.
    When rejected, the rejection signal is recorded as feedback against the stored entry.
    """
    logger.info(f"📝 Approval request for {request.correction_id}: {'Approved' if request.approved else 'Rejected'}")

    from rag_system.chromadb_client import get_chromadb_client
    chroma_client = get_chromadb_client()

    # Resolve correction context: prefer explicit fields from request,
    # fall back to in-memory cache populated at generation time.
    cached = _pending_corrections.get(request.correction_id, {})
    test_code = request.test_code or cached.get("test_code", "")
    fix_code = request.fix_code or cached.get("fix_code", "")
    error_type = request.error_type or cached.get("error_type")
    error_message = request.error_message or cached.get("error_message", "")
    test_file = request.test_file or cached.get("test_file", "unknown")
    confidence_score = request.confidence_score if request.confidence_score is not None else cached.get("confidence_score", 0.0)

    if request.approved:
        logger.info(f"✅ Correction {request.correction_id} approved")

        rag_stored = False
        if fix_code and test_code:
            background_tasks.add_task(
                chroma_client.store_approved_fix,
                fix_id=request.correction_id,
                original_test_code=test_code,
                fix_code=fix_code,
                error_type=error_type,
                error_message=error_message or "",
                test_file=test_file,
                confidence_score=confidence_score,
            )
            rag_stored = True
            logger.info(f"   📚 Fix queued for RAG indexing (error_type={error_type})")
        else:
            logger.warning(f"   ⚠️ No fix_code/test_code available — RAG indexing skipped for {request.correction_id}")

        # Clean up cache entry.
        _pending_corrections.pop(request.correction_id, None)

        return {
            "success": True,
            "message": "Correction approved and indexed into RAG knowledge base",
            "correction_id": request.correction_id,
            "rag_indexed": rag_stored,
            "status": "approved",
        }
    else:
        logger.info(f"❌ Correction {request.correction_id} rejected")
        if request.feedback:
            logger.info(f"   Feedback: {request.feedback}")

        background_tasks.add_task(
            chroma_client.update_fix_feedback,
            fix_id=request.correction_id,
            approved=False,
            feedback=request.feedback,
        )

        _pending_corrections.pop(request.correction_id, None)

        return {
            "success": True,
            "message": "Rejection feedback recorded.",
            "correction_id": request.correction_id,
            "status": "rejected",
        }


@app.get("/stats", tags=["Stats"])
async def get_statistics():
    """
    Get usage statistics
    """
    groq_client = get_groq_client()
    stats = groq_client.get_stats()
    
    return {
        "groq_usage": {
            "tokens_used_minute": stats["tokens_used_minute"],
            "tokens_used_day": stats["tokens_used_day"],
            "rate_limit_minute": stats["rate_limit_minute"],
            "rate_limit_day": stats["rate_limit_day"],
            "percentage_used_day": round((stats["tokens_used_day"] / stats["rate_limit_day"]) * 100, 2)
        },
        "cache": {
            "size": stats["cache_size"],
            "enabled": settings.CACHE_ENABLED
        },
        "system": {
            "version": settings.APP_VERSION,
            "model": settings.GROQ_MODEL,
            "langgraph_available": LANGGRAPH_AVAILABLE
        }
    }


@app.post("/webhook/jenkins", tags=["Webhooks"])
async def jenkins_webhook(payload: Dict[Any, Any]):
    """
    Webhook endpoint for Jenkins
    
    Jenkins will call this when tests fail
    """
    logger.info("🔔 Jenkins webhook received")
    logger.debug(f"Payload: {payload}")
    
    # TODO: Phase 4 - Parse Jenkins payload and trigger analysis
    
    return {
        "success": True,
        "message": "Webhook received. Analysis will be triggered.",
        "received_at": datetime.now().isoformat()
    }


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

@app.get("/rag/stats", tags=["RAG"])
async def get_rag_stats():
    """
    Get RAG system statistics
    Returns number of indexed tests, fixes, and conventions
    """
    from rag_system.chromadb_client import get_chromadb_client
    
    chroma_client = get_chromadb_client()
    stats = chroma_client.get_stats()
    
    return {
        "chromadb": stats,
        "collections": {
            "tests": stats["tests"],
            "fixes": stats["fixes"],
            "conventions": stats["conventions"]
        },
        "total_documents": stats["total"]
    }