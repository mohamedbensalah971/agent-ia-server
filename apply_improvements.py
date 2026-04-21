#!/usr/bin/env python3
"""
Apply smart change analysis to the generate_tests endpoint.
Run this script to update main.py with the file change analyzer integration.
"""

import sys
from pathlib import Path

def apply_smart_change_analysis():
    """Add smart change analysis logic to generate_tests endpoint."""
    
    main_py_path = Path(__file__).parent / "main.py"
    
    if not main_py_path.exists():
        print(f"❌ Could not find main.py at {main_py_path}")
        return False
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Check if already applied
    if "FileChangeAnalyzer" in content:
        print("⚙️ Smart change analysis already applied!")
        return True
    
    # Find the insertion point: right after "logger.info(f"🧪 [GEN]...")"
    marker = 'logger.info(f"🧪 [GEN] Received test generation request for: {request.source_file}")'
    if marker not in content:
        print(f"❌ Could not find insertion point in main.py")
        print(f"   Looking for: {marker}")
        return False
    
    # The code to insert
    insertion_code = '''

    try:
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: SMART CHANGE ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        
        if request.analyze_changes and request.original_code:
            from file_change_analyzer import FileChangeAnalyzer
            
            requires_tests, analysis_reason = FileChangeAnalyzer.requires_test_generation(
                request.source_code,
                request.original_code
            )
            
            if not requires_tests:
                logger.info(f"⏭️ Skipping test generation: {analysis_reason}")
                return TestGenerationResponse(
                    success=True,
                    generation_id=f"skip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    generated_tests="",
                    quality_notes=[f"✅ No test generation needed: {analysis_reason}"],
                    explanation=analysis_reason,
                )
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: RAG CONTEXT RETRIEVAL
        # ═══════════════════════════════════════════════════════════════════
'''

    # Find where to replace "try:" in the function
    target_try = 'try:\n        groq_client = get_groq_client()'
    
    if target_try in content:
        new_content = content.replace(target_try, insertion_code)
        
        with open(main_py_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Smart change analysis integrated into main.py")
        print("   Added:Locations:")
        print("   - Smart change detection before RAG context")
        print("   - Skip generation for cosmetic changes")
        print("   - Clear logging of why generation was skipped")
        return True
    else:
        print("❌ Could not find 'try: groq_client' pattern in main.py")
        print("   This might mean the file structure has changed.")
        print("   Manual integration may be required.")
        return False


def verify_file_analyzer():
    """Verify that file_change_analyzer.py exists."""
    analyzer_path = Path(__file__).parent / "file_change_analyzer.py"
    
    if analyzer_path.exists():
        print("✅ file_change_analyzer.py exists")
        return True
    else:
        print("❌ file_change_analyzer.py not found!")
        return False


def verify_groq_client_enhanced():
    """Verify that groq_client.py has been enhanced."""
    groq_path = Path(__file__).parent / "groq_client.py"
    
    if not groq_path.exists():
        print("❌ groq_client.py not found!")
        return False
    
    with open(groq_path, 'r') as f:
        content = f.read()
    
    # Check for enhanced prompt indicators
    if "COMMON ANNOTATION MISTAKES TO AVOID" in content and "❌ @BeforeEachEach" in content:
        print("✅ groq_client.py has enhanced system prompt")
        return True
    else:
        print("⚠️ groq_client.py might not have been fully enhanced")
        print("   Check that _get_test_generation_system_prompt() contains new examples")
        return False


def verify_main_py_updated():
    """Verify that main.py TestGenerationRequest has been updated."""
    main_py_path = Path(__file__).parent / "main.py"
    
    if not main_py_path.exists():
        print("❌ main.py not found!")
        return False
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Check for new fields
    if "original_code:" in content and "analyze_changes:" in content:
        print("✅ main.py TestGenerationRequest has been updated")
        return True
    else:
        print("⚠️ main.py TestGenerationRequest fields might not be updated")
        return False


def main():
    """Run all verification and integration steps."""
    print()
    print("=" * 70)
    print("SMART CHANGE ANALYSIS - INTEGRATION VERIFICATION")
    print("=" * 70)
    print()
    
    all_good = True
    
    print("1. Checking file_change_analyzer.py...")
    if not verify_file_analyzer():
        all_good = False
    print()
    
    print("2. Checking groq_client.py enhancements...")
    if not verify_groq_client_enhanced():
        all_good = False
    print()
    
    print("3. Checking main.py updates...")
    if not verify_main_py_updated():
        all_good = False
    print()
    
    print("4. Applying smart change analysis to generate_tests endpoint...")
    if not apply_smart_change_analysis():
        all_good = False
    print()
    
    if all_good:
        print("=" * 70)
        print("✅ SUCCESS! All improvements have been integrated.")
        print("=" * 70)
        print()
        print("     Your agent is now enhanced with:")
        print("     • Smart change detection (skips tests for cosmetic changes)")
        print("     • Improved prompts (better @BeforeEach/@AfterEach guidance)")
        print("     • Better validation (catches typos early)")
        print()
        print("     Next: Run the agent server and test with:")
        print("     curl -X POST http://localhost:8000/generate-tests \\")
        print("       -H 'Content-Type: application/json' \\")
        print("       -d '{...your request...}'")
        print()
        return 0
    else:
        print("=" * 70)
        print("⚠️ PARTIAL SUCCESS - Some checks failed or skipped")
        print("=" * 70)
        print()
        print("Please review the messages above and fix any issues.")
        print("You may need to manually integrate some changes.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
