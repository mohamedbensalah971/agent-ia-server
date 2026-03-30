"""
Groq API Client
Handles communication with Groq Llama 3.3 70B API
Includes rate limiting and caching
"""

from groq import Groq
from typing import Optional, Dict, Any, List
import hashlib
import json
from datetime import datetime, timedelta
from cachetools import TTLCache
from loguru import logger

from config import settings


class GroqClient:
    """
    Client for Groq API with rate limiting and caching
    """
    
    def __init__(self):
        """Initialize Groq client"""
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        
        # Cache for similar queries (MD5 hash -> response)
        self.cache = TTLCache(maxsize=100, ttl=settings.CACHE_TTL)
        
        # Rate limiting tracking
        self.tokens_used_minute = 0
        self.tokens_used_day = 0
        self.minute_reset = datetime.now() + timedelta(minutes=1)
        self.day_reset = datetime.now() + timedelta(days=1)
        
        logger.info(f"✅ Groq client initialized with model: {self.model}")
    
    def _reset_rate_limits(self):
        """Reset rate limit counters if time window passed"""
        now = datetime.now()
        
        if now >= self.minute_reset:
            self.tokens_used_minute = 0
            self.minute_reset = now + timedelta(minutes=1)
            logger.debug("Rate limit minute counter reset")
        
        if now >= self.day_reset:
            self.tokens_used_day = 0
            self.day_reset = now + timedelta(days=1)
            logger.info("Rate limit day counter reset")
    
    def _check_rate_limits(self, estimated_tokens: int) -> bool:
        """
        Check if request would exceed rate limits
        
        Args:
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            True if within limits, False otherwise
        """
        self._reset_rate_limits()
        
        if (self.tokens_used_minute + estimated_tokens > settings.RATE_LIMIT_TOKENS_PER_MINUTE):
            logger.warning(f"⚠️ Rate limit reached: {self.tokens_used_minute}/{settings.RATE_LIMIT_TOKENS_PER_MINUTE} tokens/minute")
            return False
        
        if (self.tokens_used_day + estimated_tokens > settings.RATE_LIMIT_TOKENS_PER_DAY):
            logger.warning(f"⚠️ Daily rate limit reached: {self.tokens_used_day}/{settings.RATE_LIMIT_TOKENS_PER_DAY} tokens/day")
            return False
        
        return True
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate cache key from prompt
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            MD5 hash as cache key
        """
        content = f"{system_prompt}:{prompt}" if system_prompt else prompt
        return hashlib.md5(content.encode()).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens (1 token ≈ 4 characters)
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def generate_correction(
        self,
        test_code: str,
        error_logs: str,
        source_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate correction for failing test
        
        Args:
            test_code: The failing test code
            error_logs: Error logs from test execution
            source_code: Optional source code being tested
            context: Optional additional context (from RAG later)
            
        Returns:
            Dict with correction and metadata
        """
        # Build prompt
        prompt = self._build_correction_prompt(test_code, error_logs, source_code, context)
        
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        if settings.CACHE_ENABLED and cache_key in self.cache:
            logger.info(f"✅ Cache hit for correction (saved tokens!)")
            return self.cache[cache_key]
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(prompt) + settings.GROQ_MAX_TOKENS
        
        # Check rate limits
        if not self._check_rate_limits(estimated_tokens):
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "rate_limit_exceeded": True
            }
        
        try:
            logger.info(f"🔄 Calling Groq API (estimated: {estimated_tokens} tokens)...")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            
            # Extract response
            correction_text = response.choices[0].message.content
            
            # Update token usage
            tokens_used = response.usage.total_tokens
            self.tokens_used_minute += tokens_used
            self.tokens_used_day += tokens_used
            
            logger.info(f"✅ Groq API success ({tokens_used} tokens used)")
            logger.debug(f"Tokens today: {self.tokens_used_day}/{settings.RATE_LIMIT_TOKENS_PER_DAY}")
            
            # Parse response
            result = self._parse_correction_response(correction_text)
            result["tokens_used"] = tokens_used
            result["success"] = True
            
            # Cache the result
            if settings.CACHE_ENABLED:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def generate_unit_tests(
        self,
        source_code: str,
        class_name: Optional[str] = None,
        existing_tests: Optional[str] = None,
        framework: str = "junit5_mockk",
        include_edge_cases: bool = True,
        max_tests: int = 6,
        rag_context: Optional[str] = None,
        test_target: str = "pure_unit",
        auto_repair: bool = True,
        max_repair_attempts: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate Kotlin unit tests from source code with validation and auto-repair.
        
        Args:
            source_code: The source code to generate tests for
            class_name: Optional target class name
            existing_tests: Optional existing tests to match style
            framework: Test framework (default: junit5_mockk)
            include_edge_cases: Include edge case tests
            max_tests: Maximum number of tests to generate
            rag_context: Optional RAG context
            test_target: Test type (pure_unit or android_ui)
            auto_repair: Automatically fix validation errors (default: True)
            max_repair_attempts: Max attempts to repair (default: 2)
            
        Returns:
            Dict with generated_tests, validation_errors, success status
        """
        prompt = self._build_test_generation_prompt(
            source_code=source_code,
            class_name=class_name,
            existing_tests=existing_tests,
            framework=framework,
            include_edge_cases=include_edge_cases,
            max_tests=max_tests,
            rag_context=rag_context,
            test_target=test_target,
        )

        cache_key = self._get_cache_key(
            prompt,
            system_prompt=self._get_test_generation_system_prompt(),
        )
        if settings.CACHE_ENABLED and cache_key in self.cache:
            logger.info("✅ Cache hit for test generation")
            return self.cache[cache_key]

        estimated_tokens = self._estimate_tokens(prompt) + settings.GROQ_MAX_TOKENS
        if not self._check_rate_limits(estimated_tokens):
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "rate_limit_exceeded": True,
            }

        try:
            logger.info(f"🔄 Calling Groq API for unit test generation (estimated: {estimated_tokens} tokens)...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_test_generation_system_prompt(test_target=test_target),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,  # Zero temperature for maximum determinism
                max_tokens=settings.GROQ_MAX_TOKENS,
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens
            self.tokens_used_minute += tokens_used
            self.tokens_used_day += tokens_used

            result = self._parse_test_generation_response(content)
            generated_tests = result["generated_tests"]
            
            # ═══════════════════════════════════════════════════════════════════
            # VALIDATION AND AUTO-REPAIR LOOP
            # ═══════════════════════════════════════════════════════════════════
            
            validation_errors = self._validate_generated_tests(generated_tests)
            repair_attempts = 0
            total_tokens = tokens_used
            
            while validation_errors and auto_repair and repair_attempts < max_repair_attempts:
                repair_attempts += 1
                logger.warning(f"⚠️ Found {len(validation_errors)} validation errors, attempting repair #{repair_attempts}...")
                for err in validation_errors[:5]:  # Log first 5 errors
                    logger.warning(f"   - {err}")
                
                repair_result = self.repair_generated_tests(
                    generated_tests=generated_tests,
                    validation_errors=validation_errors,
                    source_code=source_code,
                    test_target=test_target,
                )
                
                if repair_result.get("success"):
                    generated_tests = repair_result["repaired_tests"]
                    total_tokens += repair_result.get("tokens_used", 0)
                    # Re-validate after repair
                    validation_errors = self._validate_generated_tests(generated_tests)
                    if not validation_errors:
                        logger.info(f"✅ Tests repaired successfully after {repair_attempts} attempt(s)")
                else:
                    logger.warning(f"⚠️ Repair attempt #{repair_attempts} failed: {repair_result.get('error', 'Unknown error')}")
                    break
            
            # Final result
            result["generated_tests"] = generated_tests
            result["tokens_used"] = total_tokens
            result["success"] = True
            result["validation_errors"] = validation_errors
            result["repair_attempts"] = repair_attempts
            result["is_valid"] = len(validation_errors) == 0

            if validation_errors:
                logger.warning(f"⚠️ Tests generated with {len(validation_errors)} remaining validation issues")
            else:
                logger.info(f"✅ Unit tests generated and validated ({total_tokens} total tokens used)")

            if settings.CACHE_ENABLED and not validation_errors:
                # Only cache fully valid results
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"❌ Groq API error during test generation: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for the AI agent
        
        Returns:
            System prompt string
        """
        return """You are an expert Android test engineer specializing in Kotlin, JUnit 5, and MockK.

Your task is to analyze failing tests and provide precise corrections.

IMPORTANT RULES:
1. Provide ONLY the corrected code, no explanations before or after
2. Keep the same test structure and naming
3. Focus on the specific failure cause
4. Use MockK syntax correctly (mockkStatic, every, verify)
5. Handle Koin dependency injection properly
6. Return valid Kotlin code that can be directly applied

Common Android test issues you fix:
- Missing mockkStatic() for static methods
- Incorrect mock setup (every {} blocks)
- Koin initialization issues
- Dispatcher issues (Main vs IO)
- NullPointerException in mocks
- Assertion errors

Format your response as:
```kotlin
// Fixed test code here
```

Be concise and accurate."""
    
    def _build_correction_prompt(
        self,
        test_code: str,
        error_logs: str,
        source_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the correction prompt
        
        Args:
            test_code: Failing test code
            error_logs: Error logs
            source_code: Optional source code
            context: Optional context from RAG
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "# FAILING TEST",
            "```kotlin",
            test_code.strip(),
            "```",
            "",
            "# ERROR LOGS",
            "```",
            error_logs.strip()[:1000],  # Limit log size
            "```"
        ]
        
        if source_code:
            prompt_parts.extend([
                "",
                "# SOURCE CODE BEING TESTED",
                "```kotlin",
                source_code.strip()[:1000],  # Limit size
                "```"
            ])
        
        if context and context.get("similar_fixes"):
            prompt_parts.extend([
                "",
                "# SIMILAR FIXES FROM PROJECT",
                context["similar_fixes"]
            ])
        
        prompt_parts.extend([
            "",
            "# TASK",
            "Fix the test above. Return ONLY the corrected Kotlin code in a code block."
        ])
        
        return "\n".join(prompt_parts)

    def _get_test_generation_system_prompt(self, test_target: str = "pure_unit") -> str:
        """System prompt used for unit test generation requests."""
        base_rules = """You are an expert Kotlin Android test engineer. You write PRECISE, ERROR-FREE code.

TASK: Generate HIGH-QUALITY unit tests for provided source code.

═══════════════════════════════════════════════════════════════════
CRITICAL: SPELL ALL ANNOTATIONS AND KEYWORDS EXACTLY AS SHOWN BELOW
═══════════════════════════════════════════════════════════════════

CORRECT ANNOTATIONS (copy these exactly):
- @Test                    (NOT @Testt, @test, @TEST)
- @BeforeEach              (NOT @BeforeEachEach, @Beforeeach, @beforeEach)
- @AfterEach               (NOT @AfterEachEach, @Aftereach)
- @DisplayName             (NOT @Displayname, @DisplayNam)
- @Nested                  (NOT @nested, @NESTED)

CORRECT IMPORTS (copy these exactly):
```kotlin
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import io.mockk.mockk
import io.mockk.every
import io.mockk.verify
import io.mockk.slot
import io.mockk.coEvery
import io.mockk.coVerify
```

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT RULES
═══════════════════════════════════════════════════════════════════

1. Return ONLY Kotlin test code in a single ```kotlin``` block.
2. Start with package declaration, then imports, then test class.
3. Do NOT include explanations, comments about what you're doing, or markdown outside the code block.
4. Every test function must have @Test annotation on the line immediately before "fun".

═══════════════════════════════════════════════════════════════════
JUNIT 5 RULES (NEVER USE JUNIT 4)
═══════════════════════════════════════════════════════════════════

NEVER USE (JUnit 4):
- org.junit.Test
- org.junit.Before
- org.junit.After
- @RunWith
- @Rule

ALWAYS USE (JUnit 5):
- org.junit.jupiter.api.Test
- org.junit.jupiter.api.BeforeEach
- org.junit.jupiter.api.AfterEach

═══════════════════════════════════════════════════════════════════
MOCKK SYNTAX RULES
═══════════════════════════════════════════════════════════════════

CORRECT MockK patterns:
```kotlin
// Creating mocks
private lateinit var mockService: MyService
mockService = mockk()                    // relaxed=false by default
mockService = mockk(relaxed = true)      // for relaxed mocks

// Stubbing
every { mockService.getData() } returns listOf("a", "b")
every { mockService.process(any()) } returns Result.success(Unit)
coEvery { mockService.fetchAsync() } returns data  // for suspend functions

// Verification
verify { mockService.getData() }
verify(exactly = 1) { mockService.process(any()) }
coVerify { mockService.fetchAsync() }  // for suspend functions
```

═══════════════════════════════════════════════════════════════════
KOTLIN SYNTAX RULES
═══════════════════════════════════════════════════════════════════

1. Use assertEquals(expected, actual) - NOT assertEquals(actual, expected)
2. Use assertNotNull(value) - NOT assert(value != null)
3. Use assertTrue(condition) - NOT assert(condition)
4. Balance all braces: every { must have matching }
5. Balance all parentheses: every ( must have matching )
6. String literals must be closed: every " must have matching "

═══════════════════════════════════════════════════════════════════
ANDROID/KOTLIN FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════════════════

NEVER DO:
- Access private constants: ClassName.PRIVATE_CONSTANT
- Access private nested classes
- Mutate data class fields (assume all are val)
- Mock Android framework: Context, ViewGroup, View, LayoutInflater
- Use Kotlin assert() function - use JUnit assertions
- Use runTest{} unless testing suspend functions

═══════════════════════════════════════════════════════════════════
EXAMPLE OF CORRECT TEST CLASS
═══════════════════════════════════════════════════════════════════

```kotlin
package com.example.app

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Assertions.*
import io.mockk.mockk
import io.mockk.every
import io.mockk.verify

class UserServiceTest {

    private lateinit var mockRepository: UserRepository
    private lateinit var userService: UserService

    @BeforeEach
    fun setUp() {
        mockRepository = mockk(relaxed = true)
        userService = UserService(mockRepository)
    }

    @Test
    fun `getUser returns user when found`() {
        // Arrange
        val expectedUser = User(id = 1, name = "John")
        every { mockRepository.findById(1) } returns expectedUser

        // Act
        val result = userService.getUser(1)

        // Assert
        assertNotNull(result)
        assertEquals("John", result?.name)
        verify { mockRepository.findById(1) }
    }

    @Test
    fun `getUser returns null when not found`() {
        // Arrange
        every { mockRepository.findById(any()) } returns null

        // Act
        val result = userService.getUser(999)

        // Assert
        assertNull(result)
    }
}
```"""
        
        if test_target == "android_ui":
            return base_rules + """

═══════════════════════════════════════════════════════════════════
ANDROID_UI TEST SPECIFIC RULES
═══════════════════════════════════════════════════════════════════
- Use Robolectric for real view inflation if needed.
- Prefer LayoutInflater.from(context) over mocking ViewGroup.
- Test view visibility and behavior, not private internal state.
- NEVER mock ViewGroup or Context directly."""
        else:
            return base_rules + """

═══════════════════════════════════════════════════════════════════
PURE_UNIT TEST SPECIFIC RULES (default)
═══════════════════════════════════════════════════════════════════
- Do NOT mock Android framework classes (Context, ViewGroup, View, LayoutInflater).
- Test pure Kotlin logic without framework dependencies.
- Use simple mocks for business/data layer interfaces only.
- Focus on testing public methods and their return values."""

    def _build_test_generation_prompt(
        self,
        source_code: str,
        class_name: Optional[str],
        existing_tests: Optional[str],
        framework: str,
        include_edge_cases: bool,
        max_tests: int,
        rag_context: Optional[str],
        test_target: str = "pure_unit",
    ) -> str:
        """Build prompt for Kotlin unit test generation."""
        prompt_parts = [
            "═══════════════════════════════════════════════════════════════════",
            "TASK: Generate Kotlin Unit Tests",
            "═══════════════════════════════════════════════════════════════════",
            "",
            f"• Target class: {class_name or 'auto-detect from source'}",
            f"• Framework: {framework}",
            f"• Test type: {test_target}",
            f"• Number of tests: Generate up to {max_tests} tests",
            f"• Edge cases: {'Include edge cases and error scenarios' if include_edge_cases else 'Focus on happy path only'}",
            "",
            "═══════════════════════════════════════════════════════════════════",
            "SOURCE CODE TO TEST",
            "═══════════════════════════════════════════════════════════════════",
            "```kotlin",
            source_code.strip()[:5000],
            "```",
        ]

        if existing_tests:
            prompt_parts.extend([
                "",
                "═══════════════════════════════════════════════════════════════════",
                "EXISTING TESTS (match this style, avoid duplicates)",
                "═══════════════════════════════════════════════════════════════════",
                "```kotlin",
                existing_tests.strip()[:3000],
                "```",
            ])

        if rag_context:
            prompt_parts.extend([
                "",
                "═══════════════════════════════════════════════════════════════════",
                "PROJECT CONTEXT FROM KNOWLEDGE BASE",
                "═══════════════════════════════════════════════════════════════════",
                rag_context[:2500],
            ])

        prompt_parts.extend([
            "",
            "═══════════════════════════════════════════════════════════════════",
            "MANDATORY CHECKLIST (verify before responding)",
            "═══════════════════════════════════════════════════════════════════",
            "Before you output code, mentally verify:",
            "□ All @Test annotations are spelled exactly as @Test",
            "□ All @BeforeEach annotations are spelled exactly as @BeforeEach", 
            "□ All imports use org.junit.jupiter.api (NOT org.junit)",
            "□ All braces {} are balanced",
            "□ All parentheses () are balanced",
            "□ All string quotes \"\" are balanced",
            "□ No private constants or nested classes accessed",
            "□ No Android framework mocks (Context, View, ViewGroup)",
            "□ Using assertEquals(expected, actual) not assertEquals(actual, expected)",
            "",
            "═══════════════════════════════════════════════════════════════════",
            "OUTPUT",
            "═══════════════════════════════════════════════════════════════════",
            "Return ONLY a single ```kotlin``` code block containing:",
            "1. Package declaration",
            "2. Import statements", 
            "3. Test class with test methods",
            "",
            "DO NOT include any text before or after the code block.",
        ])

        return "\n".join(prompt_parts)
    
    def _parse_correction_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response to extract corrected code
        
        Args:
            response: Raw AI response
            
        Returns:
            Parsed correction data
        """
        # Extract code block
        code_start = response.find("```kotlin")
        code_end = response.find("```", code_start + 9)
        
        if code_start != -1 and code_end != -1:
            corrected_code = response[code_start + 9:code_end].strip()
        else:
            # Fallback: try to find any code block
            code_start = response.find("```")
            code_end = response.find("```", code_start + 3)
            if code_start != -1 and code_end != -1:
                corrected_code = response[code_start + 3:code_end].strip()
            else:
                # No code block found, use entire response
                corrected_code = response.strip()
        
        return {
            "corrected_code": corrected_code,
            "explanation": self._extract_explanation(response),
            "confidence": self._estimate_confidence(response)
        }

    def _parse_test_generation_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response to extract generated unit test code."""
        code_start = response.find("```kotlin")
        code_end = response.find("```", code_start + 9)

        if code_start != -1 and code_end != -1:
            generated_tests = response[code_start + 9:code_end].strip()
        else:
            code_start = response.find("```")
            code_end = response.find("```", code_start + 3)
            if code_start != -1 and code_end != -1:
                generated_tests = response[code_start + 3:code_end].strip()
            else:
                generated_tests = response.strip()

        return {
            "generated_tests": generated_tests,
            "explanation": self._extract_explanation(response),
            "confidence": self._estimate_confidence(response),
        }

    def _validate_generated_tests(self, code: str) -> List[str]:
        """
        Comprehensive validation of generated test code.
        Catches typos, syntax issues, and common mistakes.
        
        Args:
            code: Generated Kotlin test code
            
        Returns:
            List of validation error messages (empty if valid)
        """
        import re
        errors = []
        
        if not code or not code.strip():
            return ["Empty code generated"]
        
        # ═══════════════════════════════════════════════════════════════════
        # ANNOTATION TYPO DETECTION
        # ═══════════════════════════════════════════════════════════════════
        
        # Common @Test typos
        test_typos = re.findall(r'@[Tt][Ee][Ss][Tt]+(?![a-zA-Z])', code)
        for typo in test_typos:
            if typo != "@Test":
                errors.append(f"Annotation typo: '{typo}' should be '@Test'")
        
        # Common @BeforeEach typos  
        before_typos = re.findall(r'@[Bb]efore[Ee]ach[Ee]?a?c?h?', code)
        for typo in before_typos:
            if typo != "@BeforeEach":
                errors.append(f"Annotation typo: '{typo}' should be '@BeforeEach'")
        
        # Common @AfterEach typos
        after_typos = re.findall(r'@[Aa]fter[Ee]ach[Ee]?a?c?h?', code)
        for typo in after_typos:
            if typo != "@AfterEach":
                errors.append(f"Annotation typo: '{typo}' should be '@AfterEach'")
        
        # Detect doubled annotations like @TestTest, @BeforeEachEach
        doubled_annotations = re.findall(r'@(\w+)\1', code)
        for match in doubled_annotations:
            errors.append(f"Doubled annotation detected: '@{match}{match}' - remove duplicate")
        
        # Detect other common annotation misspellings
        if re.search(r'@Displayname\b', code):  # lowercase 'n'
            errors.append("Annotation typo: '@Displayname' should be '@DisplayName'")
        if re.search(r'@displayName\b', code):  # lowercase 'd'
            errors.append("Annotation typo: '@displayName' should be '@DisplayName'")
        if re.search(r'@nested\b', code):  # lowercase
            errors.append("Annotation typo: '@nested' should be '@Nested'")
        
        # ═══════════════════════════════════════════════════════════════════
        # IMPORT VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
        # Check for JUnit 4 imports (forbidden)
        if "org.junit.Test" in code and "org.junit.jupiter" not in code:
            errors.append("JUnit 4 import detected: 'org.junit.Test' - use 'org.junit.jupiter.api.Test'")
        if "org.junit.Before" in code and "jupiter" not in code:
            errors.append("JUnit 4 import detected: 'org.junit.Before' - use 'org.junit.jupiter.api.BeforeEach'")
        if "org.junit.After" in code and "jupiter" not in code:
            errors.append("JUnit 4 import detected: 'org.junit.After' - use 'org.junit.jupiter.api.AfterEach'")
        if "@RunWith" in code:
            errors.append("JUnit 4 annotation '@RunWith' detected - not compatible with JUnit 5")
        if "@Rule" in code and "@JvmField" not in code:
            errors.append("JUnit 4 annotation '@Rule' detected - use JUnit 5 extensions instead")
        
        # ═══════════════════════════════════════════════════════════════════
        # SYNTAX BALANCE CHECKS
        # ═══════════════════════════════════════════════════════════════════
        
        # Brace balance
        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} '{{' vs {close_braces} '}}'")
        
        # Parenthesis balance
        open_parens = code.count("(")
        close_parens = code.count(")")
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} '(' vs {close_parens} ')'")
        
        # Bracket balance
        open_brackets = code.count("[")
        close_brackets = code.count("]")
        if open_brackets != close_brackets:
            errors.append(f"Unbalanced brackets: {open_brackets} '[' vs {close_brackets} ']'")
        
        # String quote balance (rough check - may have false positives with escaped quotes)
        # Count quotes that aren't escaped and aren't in triple quotes
        single_quotes = len(re.findall(r'(?<!\\)(?<!\')"(?!"")(?!"")', code))
        if single_quotes % 2 != 0:
            errors.append("Potentially unbalanced double quotes")
        
        # ═══════════════════════════════════════════════════════════════════
        # STRUCTURE VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
        # Must have at least one @Test annotation
        if "@Test" not in code:
            errors.append("No @Test annotation found - tests must have @Test")
        
        # Must have at least one 'fun ' declaration
        if "fun " not in code:
            errors.append("No function declaration found - missing 'fun '")
        
        # Check for class declaration
        if "class " not in code:
            errors.append("No class declaration found - test must be in a class")
        
        # ═══════════════════════════════════════════════════════════════════
        # FORBIDDEN PATTERNS
        # ═══════════════════════════════════════════════════════════════════
        
        # Using Kotlin assert() instead of JUnit assertions
        if re.search(r'\bassert\s*\(', code) and 'assertEquals' not in code:
            errors.append("Using Kotlin 'assert()' - use JUnit assertions (assertEquals, assertTrue, etc.)")
        
        # Accessing private constants (pattern: ClassName.CONSTANT_NAME where CONSTANT is all caps)
        private_const_access = re.findall(r'\b[A-Z][a-zA-Z0-9]+\.([A-Z][A-Z_0-9]+)\b', code)
        # Filter out known valid patterns
        known_valid = {'MAX_VALUE', 'MIN_VALUE', 'POSITIVE_INFINITY', 'NEGATIVE_INFINITY', 'NaN'}
        for const in private_const_access:
            if const not in known_valid:
                # Only warn if it looks like an internal constant
                if const.startswith('VIEW_TYPE_') or const.startswith('TYPE_') or '_ID' in const:
                    errors.append(f"Possible private constant access: '{const}' - test public API instead")
        
        # Mocking Android framework classes
        android_mock_patterns = [
            (r'mockk<Context>', "Mocking Context is forbidden in pure unit tests"),
            (r'mockk<ViewGroup>', "Mocking ViewGroup is forbidden in pure unit tests"),
            (r'mockk<View>', "Mocking View is forbidden in pure unit tests"),
            (r'mockk<LayoutInflater>', "Mocking LayoutInflater is forbidden in pure unit tests"),
            (r'mock\(Context::class', "Mocking Context is forbidden in pure unit tests"),
            (r'mock\(ViewGroup::class', "Mocking ViewGroup is forbidden in pure unit tests"),
        ]
        for pattern, message in android_mock_patterns:
            if re.search(pattern, code):
                errors.append(message)
        
        # ═══════════════════════════════════════════════════════════════════
        # MOCKK SYNTAX VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
        # Check for common MockK mistakes
        if "every{" in code.replace(" ", ""):
            # Check if there's a space issue: every{ vs every {
            if re.search(r'every\{[^}]', code):
                errors.append("MockK syntax: 'every{' should have space: 'every {'")
        
        if "verify{" in code.replace(" ", ""):
            if re.search(r'verify\{[^}]', code):
                errors.append("MockK syntax: 'verify{' should have space: 'verify {'")
        
        # ═══════════════════════════════════════════════════════════════════
        # COMMON TYPOS IN KOTLIN KEYWORDS
        # ═══════════════════════════════════════════════════════════════════
        
        keyword_typos = [
            (r'\bfunn\b', 'funn', 'fun'),
            (r'\boverridee\b', 'overridee', 'override'),
            (r'\bretuns\b', 'retuns', 'returns'),
            (r'\bprivatte\b', 'privatte', 'private'),
            (r'\blateinitt\b', 'lateinitt', 'lateinit'),
            (r'\bassertEqualss\b', 'assertEqualss', 'assertEquals'),
            (r'\bassertTruee\b', 'assertTruee', 'assertTrue'),
            (r'\bassertFalsee\b', 'assertFalsee', 'assertFalse'),
            (r'\bassertNotNulll\b', 'assertNotNulll', 'assertNotNull'),
            (r'\bassertNulll\b', 'assertNulll', 'assertNull'),
        ]
        
        for pattern, typo, correct in keyword_typos:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Keyword typo: '{typo}' should be '{correct}'")
        
        return errors
    
    def repair_generated_tests(
        self,
        generated_tests: str,
        validation_errors: List[str],
        source_code: Optional[str] = None,
        test_target: str = "pure_unit",
    ) -> Dict[str, Any]:
        """
        Auto-fix generated tests that fail validation.
        
        Args:
            generated_tests: Previously generated test code with errors
            validation_errors: List of validation errors found
            source_code: Optional source code being tested
            test_target: Target test type (pure_unit or android_ui)
            
        Returns:
            Dict with repaired_tests, explanation, success status
        """
        if not validation_errors:
            return {"success": True, "repaired_tests": generated_tests, "changes": []}
        
        errors_summary = "\n".join([f"• {err}" for err in validation_errors[:10]])
        
        repair_prompt = f"""═══════════════════════════════════════════════════════════════════
TASK: FIX VALIDATION ERRORS IN TEST CODE
═══════════════════════════════════════════════════════════════════

You must fix the validation errors listed below. Return ONLY the corrected Kotlin code.

═══════════════════════════════════════════════════════════════════
VALIDATION ERRORS TO FIX
═══════════════════════════════════════════════════════════════════
{errors_summary}

═══════════════════════════════════════════════════════════════════
ORIGINAL TEST CODE (with errors)
═══════════════════════════════════════════════════════════════════
```kotlin
{generated_tests.strip()[:3500]}
```

═══════════════════════════════════════════════════════════════════
CORRECTION RULES
═══════════════════════════════════════════════════════════════════

ANNOTATION SPELLING (copy exactly):
• @Test (NOT @Testt, @test)
• @BeforeEach (NOT @BeforeEachEach, @Beforeeach)
• @AfterEach (NOT @AfterEachEach)
• @DisplayName (NOT @Displayname)

REQUIRED FIXES:
1. Fix ALL typos in annotations and keywords
2. Balance all braces {{}}, parentheses (), brackets []
3. Use JUnit 5 imports: org.junit.jupiter.api.*
4. Remove any Android framework mocks (Context, View, ViewGroup)
5. Remove any private constant access (ClassName.PRIVATE_CONST)

FORBIDDEN:
• JUnit 4 imports (org.junit.Test, org.junit.Before)
• @RunWith, @Rule annotations
• Kotlin assert() function - use assertEquals, assertTrue, etc.
• Mocking Context, View, ViewGroup, LayoutInflater

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY the corrected code in a single ```kotlin``` block.
Do NOT include explanations or comments outside the code block."""
        
        if source_code:
            repair_prompt += f"""

═══════════════════════════════════════════════════════════════════
SOURCE CODE (for reference only)
═══════════════════════════════════════════════════════════════════
```kotlin
{source_code.strip()[:1500]}
```"""

        estimated_tokens = self._estimate_tokens(repair_prompt) + settings.GROQ_MAX_TOKENS
        if not self._check_rate_limits(estimated_tokens):
            return {
                "success": False,
                "error": "Rate limit exceeded during repair",
                "rate_limit_exceeded": True,
            }

        try:
            logger.info(f"🔧 Attempting to repair generated tests with {len(validation_errors)} validation errors...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_test_generation_system_prompt(test_target=test_target),
                    },
                    {
                        "role": "user",
                        "content": repair_prompt,
                    },
                ],
                temperature=0.1,  # Slightly higher than generation to allow fixes but stay deterministic
                max_tokens=settings.GROQ_MAX_TOKENS,
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens
            self.tokens_used_minute += tokens_used
            self.tokens_used_day += tokens_used

            repaired_tests = self._parse_test_generation_response(content)["generated_tests"]
            
            logger.info(f"✅ Test repair complete ({tokens_used} tokens used)")
            return {
                "success": True,
                "repaired_tests": repaired_tests,
                "tokens_used": tokens_used,
                "errors_fixed": len(validation_errors),
            }

        except Exception as e:
            logger.error(f"❌ Error during test repair: {e}")
            return {
                "success": False,
                "error": str(e),
                "repaired_tests": generated_tests,  # Return original if repair fails
            }
    
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response (if any)"""
        # Look for explanation before code block
        code_start = response.find("```")
        if code_start > 10:
            return response[:code_start].strip()
        return ""
    
    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence based on response characteristics
        
        Args:
            response: AI response
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.8  # Base confidence
        
        # Higher confidence if code block is present
        if "```kotlin" in response:
            confidence += 0.1
        
        # Lower confidence if response is very short
        if len(response) < 50:
            confidence -= 0.2
        
        # Lower confidence if contains uncertain language
        uncertain_words = ["maybe", "might", "could", "possibly", "not sure"]
        if any(word in response.lower() for word in uncertain_words):
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Returns:
            Dict with stats
        """
        self._reset_rate_limits()
        return {
            "tokens_used_minute": self.tokens_used_minute,
            "tokens_used_day": self.tokens_used_day,
            "cache_size": len(self.cache),
            "rate_limit_minute": settings.RATE_LIMIT_TOKENS_PER_MINUTE,
            "rate_limit_day": settings.RATE_LIMIT_TOKENS_PER_DAY,
            "minute_remaining": (self.minute_reset - datetime.now()).total_seconds(),
            "day_remaining": (self.day_reset - datetime.now()).total_seconds()
        }


# Global client instance
_groq_client: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """
    Get or create Groq client instance (singleton)
    
    Returns:
        GroqClient instance
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


if __name__ == "__main__":
    # Test the client
    from config import validate_settings
    
    validate_settings()
    
    client = get_groq_client()
    
    # Test with a simple example
    test_code = """
    @Test
    fun testUserLogin() {
        val user = userManager.login("test@example.com", "password")
        assertEquals("test@example.com", user.email)
    }
    """
    
    error_logs = """
    java.lang.NullPointerException: userManager is null
    at com.smarttalk.UserManagerTest.testUserLogin(UserManagerTest.kt:25)
    """
    
    print("Testing Groq client...")
    result = client.generate_correction(test_code, error_logs)
    
    if result["success"]:
        print("✅ Correction generated:")
        print(result["corrected_code"])
        print(f"Confidence: {result['confidence']}")
        print(f"Tokens used: {result['tokens_used']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n📊 Stats:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
