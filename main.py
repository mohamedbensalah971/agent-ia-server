"""
AI Agent Server - Main FastAPI Application
Supports both direct Groq calls and LangGraph workflow
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
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


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    groq_api: str
    langgraph_available: bool
    tokens_available: Dict[str, Any]


# ==========================================
# STARTUP & SHUTDOWN
# ==========================================

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
            "langgraph": "/analyze-failure-langgraph" if LANGGRAPH_AVAILABLE else "Not available"
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
        logger.info("🔄 Creating LangGraph workflow...")
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
    Approve or reject a correction
    
    If approved, the correction will be applied to Git (Phase 3)
    """
    logger.info(f"📝 Approval request for {request.correction_id}: {'Approved' if request.approved else 'Rejected'}")
    
    if request.approved:
        # TODO: Phase 3 - Apply correction to Git
        logger.info(f"✅ Correction {request.correction_id} approved")
        logger.info("   [Phase 3] Will apply to Git and create branch")
        
        return {
            "success": True,
            "message": "Correction approved and queued for application",
            "correction_id": request.correction_id,
            "status": "pending_git_application"
        }
    else:
        logger.info(f"❌ Correction {request.correction_id} rejected")
        if request.feedback:
            logger.info(f"   Feedback: {request.feedback}")
        
        # TODO: Store rejection feedback for AI learning
        
        return {
            "success": True,
            "message": "Correction rejected. Feedback recorded.",
            "correction_id": request.correction_id,
            "status": "rejected"
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