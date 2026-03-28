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
    ) -> Dict[str, Any]:
        """Generate Kotlin unit tests from source code."""
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
                temperature=0.05,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens
            self.tokens_used_minute += tokens_used
            self.tokens_used_day += tokens_used

            result = self._parse_test_generation_response(content)
            result["tokens_used"] = tokens_used
            result["success"] = True

            if settings.CACHE_ENABLED:
                self.cache[cache_key] = result

            logger.info(f"✅ Unit tests generated ({tokens_used} tokens used)")
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
        base_rules = """You are an expert Kotlin Android test engineer.

Your task is to generate HIGH-QUALITY unit tests for provided source code.

CORE RULES:
1. Return ONLY Kotlin test code in a single ```kotlin``` block.
2. Use JUnit 5 (org.junit.jupiter.*) and MockK patterns when applicable.
3. Cover happy path, error path, and edge cases when possible.
4. Keep generated tests deterministic and readable.
5. Include setup/teardown only if needed.
6. Do not include explanations outside the code block.
7. Do NOT use JUnit4 annotations/classes (RunWith, JUnit4, Rule, org.junit.Test, org.junit.Before).
8. Avoid coroutine test wrappers (runTest/TestCoroutineRule) unless the source API is suspend/Flow/coroutine-based.
9. Prefer assertTrue/assertFalse/assertEquals/assertNotNull from JUnit 5 assertions, not Kotlin assert(...).

CRITICAL ANDROID TEST RULES:
10. NEVER access private member constants or private nested classes (e.g., never use ClassName.PRIVATE_CONSTANT or PrivateNestedClass).
11. NEVER mutate model fields in tests—assume all model fields are val (immutable). Test public behavior only.
12. NEVER mock Context/ViewGroup for view inflation tests. Use only public adapter methods (like getItemViewType).
13. For RecyclerView adapters: test getItemViewType(position), assertions on returned values, not internal constants.
14. Ensure all annotation names are spelled correctly (@BeforeEach, not @BeforeEachEach; @Test, not @Testt).
15. Never reference private companion object constants. Test public method outputs instead.
16. If you cannot generate a valid test respecting these rules, return empty code block."""
        
        if test_target == "android_ui":
            return base_rules + """

FOR ANDROID_UI TESTS:
- Use Robolectric for real view inflation if needed.
- Prefer LayoutInflater.from(context) over mocking ViewGroup.
- Test view visibility and behavior, not private internal state."""
        else:
            return base_rules + """

FOR PURE_UNIT TESTS (default):
- Do NOT mock Android framework classes (Context, ViewGroup, View, LayoutInflater, etc.).
- Test pure Kotlin logic without framework dependencies.
- Use simple mocks for business logic only."""

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
            "# TASK",
            "Generate Kotlin unit tests for the provided source code.",
            f"Framework: {framework}",
            f"Target class: {class_name or 'auto-detect'}",
            f"Test type: {test_target}",
            f"Max number of tests: {max_tests}",
            f"Include edge cases: {'yes' if include_edge_cases else 'no'}",
            "",
            "# SOURCE CODE",
            "```kotlin",
            source_code.strip()[:5000],
            "```",
        ]

        if existing_tests:
            prompt_parts.extend([
                "",
                "# EXISTING TESTS (avoid duplicates, follow style)",
                "```kotlin",
                existing_tests.strip()[:3000],
                "```",
            ])

        if rag_context:
            prompt_parts.extend([
                "",
                "# PROJECT CONTEXT FROM RAG",
                rag_context[:2500],
            ])

        prompt_parts.extend([
            "",
            "# OUTPUT FORMAT",
            "Return ONLY one Kotlin code block with the generated tests.",
            "Constraints: use JUnit5 imports, avoid JUnit4/RunWith/Rule, avoid runTest unless source clearly requires coroutines.",
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
        
        errors_summary = "\n".join([f"- {err}" for err in validation_errors[:10]])  # Limit to 10 errors
        
        repair_prompt = f"""You are an expert Kotlin Android test engineer.

TASK: Fix the following test code to resolve validation errors. Return ONLY the corrected Kotlin code.

ORIGINAL TEST CODE:
```kotlin
{generated_tests.strip()[:3000]}
```

VALIDATION ERRORS TO FIX:
{errors_summary}

REQUIREMENTS:
1. Fix ONLY the listed errors without changing test intent
2. Do NOT mock Android framework classes (Context, ViewGroup, View, ViewGroup, etc.)
3. Do NOT access private constants (ClassName.CONSTANT_NAME)
4. Do NOT mutate model fields (assume all fields are val)
5. Ensure all annotations are correctly spelled (@BeforeEach, @Test, not typos)
6. Return ONLY the corrected Kotlin code in a ```kotlin``` block
7. Preserve test method names and structure"""
        
        if source_code:
            repair_prompt += f"""

SOURCE CODE BEING TESTED (for reference):
```kotlin
{source_code.strip()[:2000]}
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
