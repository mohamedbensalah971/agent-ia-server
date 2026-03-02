"""
Quick Test Script for AI Agent Server
Tests the basic functionality without running the full server
"""

import sys
import os

print("=" * 60)
print("🧪 TESTING AI AGENT SERVER - PHASE 1")
print("=" * 60)

# Test 1: Python version
print("\n1️⃣ Checking Python version...")
if sys.version_info >= (3, 11):
    print(f"   ✅ Python {sys.version.split()[0]} (OK)")
else:
    print(f"   ❌ Python {sys.version.split()[0]} (Need 3.11+)")
    sys.exit(1)

# Test 2: Import dependencies
print("\n2️⃣ Checking dependencies...")
try:
    import fastapi
    print(f"   ✅ FastAPI {fastapi.__version__}")
except ImportError:
    print("   ❌ FastAPI not installed")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import groq
    print(f"   ✅ Groq SDK installed")
except ImportError:
    print("   ❌ Groq SDK not installed")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from loguru import logger
    print(f"   ✅ Loguru installed")
except ImportError:
    print("   ❌ Loguru not installed")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check .env file
print("\n3️⃣ Checking configuration...")
if os.path.exists(".env"):
    print("   ✅ .env file found")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    if os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key.startswith("gsk_"):
            print("   ✅ GROQ_API_KEY configured")
        else:
            print("   ⚠️  GROQ_API_KEY format looks wrong (should start with gsk_)")
    else:
        print("   ❌ GROQ_API_KEY not set in .env")
        print("   Add: GROQ_API_KEY=your_key_here")
    
    if os.getenv("GIT_REPO_PATH"):
        repo_path = os.getenv("GIT_REPO_PATH")
        if os.path.exists(repo_path):
            print(f"   ✅ Git repo found at: {repo_path}")
        else:
            print(f"   ⚠️  Git repo not found at: {repo_path}")
    else:
        print("   ⚠️  GIT_REPO_PATH not set in .env")
else:
    print("   ❌ .env file not found")
    print("   Copy .env.example to .env and configure it")

# Test 4: Check logs directory
print("\n4️⃣ Checking logs directory...")
if os.path.exists("logs"):
    print("   ✅ logs/ directory exists")
else:
    print("   ⚠️  logs/ directory missing")
    print("   Creating it now...")
    os.makedirs("logs")
    print("   ✅ logs/ directory created")

# Test 5: Test Groq client (if API key is set)
print("\n5️⃣ Testing Groq API connection...")
try:
    from config import settings
    from groq_client import get_groq_client
    
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY.startswith("gsk_"):
        client = get_groq_client()
        
        # Simple test
        test_result = client.generate_correction(
            test_code="@Test\nfun simpleTest() { assertTrue(true) }",
            error_logs="Test passed (this is just a test)",
        )
        
        if test_result.get("success") or test_result.get("corrected_code"):
            print("   ✅ Groq API connected and working!")
            print(f"   📊 Tokens used: {test_result.get('tokens_used', 'N/A')}")
        else:
            print("   ⚠️  Groq API responded but result unexpected")
            print(f"   Response: {test_result}")
    else:
        print("   ⏭️  Skipping (no valid API key)")
        
except Exception as e:
    print(f"   ⚠️  Groq test failed: {e}")
    print("   This might be OK if you haven't configured .env yet")

# Summary
print("\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)
print("\nIf all tests passed, you can start the server with:")
print("   python main.py")
print("\nThen visit:")
print("   http://localhost:8000/docs")
print("=" * 60)
