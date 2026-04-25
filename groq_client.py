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
import re as regex_module


def analyze_kotlin_code(source_code: str) -> Dict[str, Any]:
    """
    Analyze Kotlin source code to detect testability constraints.
    
    This helps the LLM understand what CAN and CANNOT be tested directly,
    preventing logic errors in generated tests.
    
    Args:
        source_code: The Kotlin source code to analyze
        
    Returns:
        Dict with analysis results:
        - class_type: 'regular', 'inner', 'data', 'sealed', 'object', etc.
        - inner_classes: List of inner class names (can't instantiate directly)
        - private_classes: List of private nested class names (can't access)
        - android_dependencies: List of Android framework deps found
        - testable_methods: List of public methods that can be tested
        - constraints: List of testing constraints/warnings for the LLM
        - requires_robolectric: Boolean if Android UI testing needed
    """
    analysis = {
        "class_name": None,
        "class_type": "regular",
        "inner_classes": [],
        "private_classes": [],
        "companion_objects": [],
        "android_dependencies": [],
        "testable_methods": [],
        "private_methods": [],
        "constraints": [],
        "requires_robolectric": False,
        "extends": None,
        "implements": [],
    }
    
    # Detect main class name and type
    class_match = regex_module.search(
        r'(?:(?:public|private|internal|abstract|sealed|data|open)\s+)*'
        r'(?:inner\s+)?'
        r'class\s+(\w+)',
        source_code
    )
    if class_match:
        analysis["class_name"] = class_match.group(1)
    
    # Detect if it's a data class
    if regex_module.search(r'\bdata\s+class\b', source_code):
        analysis["class_type"] = "data"
    
    # Detect if it's a sealed class
    if regex_module.search(r'\bsealed\s+class\b', source_code):
        analysis["class_type"] = "sealed"
    
    # Detect if it's an object (singleton)
    if regex_module.search(r'\bobject\s+\w+', source_code) and 'companion object' not in source_code:
        analysis["class_type"] = "object"
    
    # Detect parent class (extends)
    extends_match = regex_module.search(
        r'class\s+\w+[^:]*:\s*(\w+)\s*[\(\{<]',
        source_code
    )
    if extends_match:
        analysis["extends"] = extends_match.group(1)
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT INNER CLASSES (Cannot be instantiated outside outer class)
    # ═══════════════════════════════════════════════════════════════════
    inner_classes = regex_module.findall(r'\binner\s+class\s+(\w+)', source_code)
    analysis["inner_classes"] = inner_classes
    for ic in inner_classes:
        analysis["constraints"].append(
            f"INNER CLASS '{ic}': Cannot instantiate directly. "
            f"Must create outer class instance first or test through outer class methods."
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT PRIVATE NESTED CLASSES (Cannot access from tests)
    # ═══════════════════════════════════════════════════════════════════
    private_classes = regex_module.findall(r'\bprivate\s+(?:inner\s+)?class\s+(\w+)', source_code)
    analysis["private_classes"] = private_classes
    for pc in private_classes:
        analysis["constraints"].append(
            f"PRIVATE CLASS '{pc}': Cannot access from tests. "
            f"Test its behavior indirectly through public API."
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT COMPANION OBJECTS
    # ═══════════════════════════════════════════════════════════════════
    if 'companion object' in source_code:
        analysis["companion_objects"].append("Companion")
        # Check for private companion members
        if regex_module.search(r'companion\s+object\s*\{[^}]*private', source_code, regex_module.DOTALL):
            analysis["constraints"].append(
                "COMPANION OBJECT has private members: Cannot access private companion members from tests."
            )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT ANDROID DEPENDENCIES
    # ═══════════════════════════════════════════════════════════════════
    android_patterns = [
        (r'\bView\b', 'View'),
        (r'\bViewGroup\b', 'ViewGroup'),
        (r'\bContext\b', 'Context'),
        (r'\bLayoutInflater\b', 'LayoutInflater'),
        (r'\bRecyclerView\b', 'RecyclerView'),
        (r'\bTextView\b', 'TextView'),
        (r'\bImageView\b', 'ImageView'),
        (r'\bButton\b', 'Button'),
        (r'\bFragment\b', 'Fragment'),
        (r'\bActivity\b', 'Activity'),
        (r'\bIntent\b', 'Intent'),
        (r'\bBundle\b', 'Bundle'),
        (r'\bSharedPreferences\b', 'SharedPreferences'),
        (r'\bLifecycleOwner\b', 'LifecycleOwner'),
        (r'\bLiveData\b', 'LiveData'),
        (r'R\.layout\.', 'R.layout'),
        (r'R\.id\.', 'R.id'),
        (r'R\.string\.', 'R.string'),
        (r'R\.drawable\.', 'R.drawable'),
        (r'\.findViewById\(', 'findViewById'),
    ]
    
    found_android = set()
    for pattern, name in android_patterns:
        if regex_module.search(pattern, source_code):
            found_android.add(name)
    
    analysis["android_dependencies"] = list(found_android)
    
    if found_android:
        # Determine if Robolectric is needed
        ui_deps = {'View', 'ViewGroup', 'TextView', 'ImageView', 'Button', 
                   'LayoutInflater', 'R.layout', 'R.id', 'findViewById'}
        if found_android & ui_deps:
            analysis["requires_robolectric"] = True
            analysis["constraints"].append(
                f"ANDROID UI DEPENDENCIES: {', '.join(found_android & ui_deps)}. "
                f"These require Robolectric for unit testing, OR test only non-UI logic."
            )
        
        lifecycle_deps = {'Fragment', 'Activity', 'LifecycleOwner'}
        if found_android & lifecycle_deps:
            analysis["constraints"].append(
                f"LIFECYCLE DEPENDENCIES: {', '.join(found_android & lifecycle_deps)}. "
                f"Mock these or use AndroidX Test libraries."
            )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT ADAPTER PATTERN
    # ═══════════════════════════════════════════════════════════════════
    if 'ListAdapter' in source_code or 'RecyclerView.Adapter' in source_code:
        analysis["class_type"] = "recycler_adapter"
        analysis["constraints"].append(
            "RECYCLERVIEW ADAPTER: ViewHolder is usually an inner class. "
            "Test DiffCallback logic directly if it's accessible. "
            "Test adapter behavior through submitList() and itemCount."
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT DIFFUTIL CALLBACK
    # ═══════════════════════════════════════════════════════════════════
    diffutil_match = regex_module.search(r'class\s+(\w+)\s*:\s*DiffUtil\.ItemCallback', source_code)
    if diffutil_match:
        callback_name = diffutil_match.group(1)
        # Check if it's private
        if regex_module.search(rf'\bprivate\s+class\s+{callback_name}', source_code):
            analysis["constraints"].append(
                f"DIFFUTIL CALLBACK '{callback_name}' is private: "
                f"Test indirectly by observing adapter behavior after submitList()."
            )
        else:
            analysis["testable_methods"].append(f"{callback_name}.areItemsTheSame()")
            analysis["testable_methods"].append(f"{callback_name}.areContentsTheSame()")
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT PUBLIC METHODS
    # ═══════════════════════════════════════════════════════════════════
    # Public methods (no private/protected modifier before fun)
    public_methods = regex_module.findall(
        r'(?<!\bprivate\s)(?<!\bprotected\s)(?<!\binternal\s)\bfun\s+(\w+)\s*\(',
        source_code
    )
    # Filter out override methods that are just implementation details
    override_methods = regex_module.findall(r'\boverride\s+fun\s+(\w+)', source_code)
    
    for method in public_methods:
        if method not in ['onCreateViewHolder', 'onBindViewHolder', 'getItemCount']:
            analysis["testable_methods"].append(method)
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECT PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════
    private_methods = regex_module.findall(r'\bprivate\s+fun\s+(\w+)', source_code)
    analysis["private_methods"] = private_methods
    if private_methods:
        analysis["constraints"].append(
            f"PRIVATE METHODS ({', '.join(private_methods[:3])}{'...' if len(private_methods) > 3 else ''}): "
            f"Test through public API, not directly."
        )
    
    return analysis


def format_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """
    Format code analysis results as clear instructions for the LLM.
    
    Args:
        analysis: Results from analyze_kotlin_code()
        
    Returns:
        Formatted string to inject into the prompt
    """
    if not analysis["constraints"]:
        return ""
    
    lines = [
        "═══════════════════════════════════════════════════════════════════",
        "⚠️  CODE ANALYSIS - TESTING CONSTRAINTS (MUST FOLLOW)",
        "═══════════════════════════════════════════════════════════════════",
    ]
    
    # Class info
    if analysis["class_name"]:
        lines.append(f"Class: {analysis['class_name']} (type: {analysis['class_type']})")
        if analysis["extends"]:
            lines.append(f"Extends: {analysis['extends']}")
    
    lines.append("")
    lines.append("CONSTRAINTS YOU MUST FOLLOW:")
    
    for i, constraint in enumerate(analysis["constraints"], 1):
        lines.append(f"{i}. {constraint}")
    
    if analysis["testable_methods"]:
        lines.append("")
        lines.append(f"TESTABLE METHODS: {', '.join(analysis['testable_methods'][:5])}")
    
    if analysis["requires_robolectric"]:
        lines.append("")
        lines.append("⚠️  ROBOLECTRIC REQUIRED: This class has Android UI dependencies.")
        lines.append("   Either use @RunWith(RobolectricTestRunner::class) OR")
        lines.append("   Only test non-UI logic (DiffCallback, data transformations, etc.)")
    
    lines.append("")
    
    return "\n".join(lines)


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
        framework: str = "junit4_mockito",
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
            
            validation_errors = self._validate_generated_tests(generated_tests, source_code=source_code)
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
                    # Re-validate after repair (with source code for context)
                    validation_errors = self._validate_generated_tests(generated_tests, source_code=source_code)
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
        base_rules = """You are an expert Kotlin Android test engineer. You write PRECISE, ERROR-FREE, PRODUCTION-QUALITY code.

TASK: Generate HIGH-QUALITY unit tests for provided source code. NO TYPOS. NO ERRORS.

═══════════════════════════════════════════════════════════════════
🚨 CRITICAL: ANNOTATION SPELLING (This is where most errors occur!)
═══════════════════════════════════════════════════════════════════

CORRECT ANNOTATIONS - COPY EXACTLY FROM HERE:
✅ @Test                  (NOT @Testt, @test, @TEST, @Testtest)
✅ @BeforeEach            (NOT @BeforeEachEach, @Beforeeach, @beforeEach, @Before)
✅ @AfterEach             (NOT @AfterEachEach, @Aftereach, @After)
✅ @DisplayName           (NOT @Displayname, @DisplayNam, @displayName)
✅ @Nested                (NOT @nested, @NESTED)
✅ @Timeout               (NOT @TimeOut, @timeout)

COMMON ANNOTATION MISTAKES TO AVOID:
❌ @BeforeEachEach        ← WRONG! Doubled word detected
❌ @TestTest              ← WRONG! Doubled word detected  
❌ @Beforeeach            ← WRONG! Lowercase 'e' in middle
❌ @beforeEach            ← WRONG! Lowercase 'b' at start
❌ @Before                ← WRONG! JUnit 4, use @BeforeEach
❌ @After                 ← WRONG! JUnit 4, use @AfterEach

RULE: Every @BeforeEach appears EXACTLY as: @BeforeEach (B-e-f-o-r-e-E-a-c-h)
RULE: Every @AfterEach appears EXACTLY as: @AfterEach (A-f-t-e-r-E-a-c-h)
RULE: Every @Test appears EXACTLY as: @Test (T-e-s-t)
RULE: NO DOUBLED ANNOTATIONS. If you write @BeforeEach once, write it exactly once.

═══════════════════════════════════════════════════════════════════
REQUIRED IMPORTS (copy this block exactly)
═══════════════════════════════════════════════════════════════════

```kotlin
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import io.mockk.mockk
import io.mockk.every
import io.mockk.verify
```

NEVER use these imports:
❌ import org.junit.Test
❌ import org.junit.Before
❌ import org.junit.After
❌ import org.junit.runner.RunWith

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT)
═══════════════════════════════════════════════════════════════════

1. Return ONLY Kotlin code in a single ```kotlin``` block. NO TEXT BEFORE OR AFTER.
2. Structure:
   - package declaration
   - imports (use the exact imports above)
   - test class with @Test methods
3. Do NOT explain your code. NO comments. NO markdown outside the code block.
4. Each test function has @Test directly on the line before @Test
   ✅ CORRECT:
   @Test
   fun myTest() { }
   
   ❌ WRONG:
   @Test fun myTest() { }  ← annotation and function on same line

═══════════════════════════════════════════════════════════════════
SYNTAX RULES (Check before returning!)
═══════════════════════════════════════════════════════════════════

BRACKETS AND BRACES:
- Every { must have matching }
- Every ( must have matching )
- Every [ must have matching ]
- VERIFY: count of { = count of }

ASSERTIONS (JUnit 5):
✅ assertEquals(expected, actual)
✅ assertNotNull(value)
✅ assertTrue(condition)
✅ assertFalse(condition)
✅ assertThrows<Exception> { code() }

❌ assert(condition)           ← Kotlin assert, not JUnit
❌ Assert.assertEquals()       ← Wrong import path
        ❌ assertEquals(actual, expected)  ← Wrong order!

        NULL SAFETY (CRITICAL for Kotlin!):
        ❌ validateMessage(null)     ← If function takes String (non-nullable), this won't compile!
        ✅ validateMessage("")       ← Use empty string to test empty input
        ✅ validateMessage("   ")    ← Use spaces to test blank input

        Only pass null if the function signature uses String? (nullable):
        fun validateMessage(input: String?) → null is OK
        fun validateMessage(input: String)  → null is FORBIDDEN, use "" instead

        FORBIDDEN LIBRARIES (not in dependencies):
        ❌ import com.google.common.truth.Truth.assertThat  ← Google Truth not available
        ✅ import org.junit.jupiter.api.Assertions.*        ← Use JUnit5 always

        REQUIRED JAVA IMPORTS (add when needed):
        ✅ import java.util.concurrent.TimeUnit    ← When using TimeUnit.MINUTES, etc.
        ✅ import java.text.SimpleDateFormat       ← When using SimpleDateFormat
        ✅ import java.util.Locale                 ← When using Locale.getDefault(), etc.
        ✅ import java.util.Date                   ← When using Date(timestamp)
        ✅ import java.util.Calendar               ← When using Calendar.getInstance()

MOCKK SETUP:
✅ every { mock.method() } returns value
✅ mockk<Service>()
✅ mockk(relaxed = true)
✅ verify { mock.method() }

❌ every{ mock.method() }      ← Missing space after 'every'
❌ every {{ }}                  ← Doubled braces
❌ mock.when().thenReturn()    ← Mockito syntax, not MockK

═══════════════════════════════════════════════════════════════════
✅ EXAMPLE OF CORRECT TEST
═══════════════════════════════════════════════════════════════════

```kotlin
package com.example

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
    fun getUserReturnsUserWhenFound() {
        val expectedUser = User(id = 1, name = "John")
        every { mockRepository.findById(1) } returns expectedUser

        val result = userService.getUser(1)

        assertNotNull(result)
        assertEquals("John", result?.name)
        verify { mockRepository.findById(1) }
    }

    @Test
    fun getUserReturnsNullWhenNotFound() {
        every { mockRepository.findById(any()) } returns null

        val result = userService.getUser(999)

        assertNull(result)
    }
}
```

═══════════════════════════════════════════════════════════════════
❌ EXAMPLES OF MISTAKES TO AVOID
═══════════════════════════════════════════════════════════════════

MISTAKE 1: Doubled annotation (most common error!)
❌ @BeforeEachEach         ← Missing 'E', doubled 'ch'
   fun setup()
   
✅ @BeforeEach             ← Correct spelling
   fun setup()

MISTAKE 2: Missing space between annotation and function
❌ @Test fun myTest() { }
✅ @Test
   fun myTest() { }

MISTAKE 3: Wrong assertion order
❌ assertEquals(actual, expected)
✅ assertEquals(expected, actual)

MISTAKE 4: Unbalanced braces
❌ every { mock.method() returns value    ← Missing }
✅ every { mock.method() } returns value

═══════════════════════════════════════════════════════════════════
ANDROID/PURE UNIT TEST RULES
═══════════════════════════════════════════════════════════════════

DO NOT MOCK (Android Framework Classes):
❌ Context
❌ ViewGroup
❌ View
❌ Activity
❌ Fragment
❌ LayoutInflater

DO MOCK (Business Logic):
✅ Repository/DAO interfaces
✅ Service interfaces
✅ API/Network clients
✅ Utility/Helper classes

═══════════════════════════════════════════════════════════════════
FINAL CHECKLIST BEFORE RETURNING CODE
═══════════════════════════════════════════════════════════════════

□ All @Test annotations are spelled: @Test (exactly)
□ All @BeforeEach annotations are spelled: @BeforeEach (not @BeforeEachEach)
□ All @AfterEach annotations are spelled: @AfterEach (not @AfterEachEach)
□ Count of { equals count of }
□ Count of ( equals count of )
□ No JUnit 4 imports (org.junit.Test, org.junit.Before, #Before, etc.)
□ No Android framework mocks (Context, ViewGroup, View)
□ No explanatory text outside the code block
□ Package and imports are complete and correct
□ Test class is declared
□ At least one @Test method exists with proper format
        □ All assertions use JUnit methods (assertEquals, assertTrue, etc.)
        □ Using assertEquals(expected, actual) not assertEquals(actual, expected)
        □ Never pass null to non-nullable String parameters (use String? if nullable needed)
        □ Never use Google Truth (com.google.common.truth) - use JUnit5 Assertions.*
        □ Always import java.util.concurrent.TimeUnit if TimeUnit is used
        □ Always import java.text.SimpleDateFormat if SimpleDateFormat is used
        □ Always import java.util.Locale if Locale is used
        □ No typos in variable names or method calls"""
        
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
        """Build prompt for Kotlin unit test generation with code analysis."""
        
        # ═══════════════════════════════════════════════════════════════════
        # ANALYZE SOURCE CODE FOR CONSTRAINTS
        # ═══════════════════════════════════════════════════════════════════
        code_analysis = analyze_kotlin_code(source_code)
        analysis_text = format_analysis_for_prompt(code_analysis)
        
        # Auto-detect class name if not provided
        if not class_name and code_analysis["class_name"]:
            class_name = code_analysis["class_name"]
        
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
        ]
        
        # ═══════════════════════════════════════════════════════════════════
        # INSERT CODE ANALYSIS CONSTRAINTS (CRITICAL!)
        # ═══════════════════════════════════════════════════════════════════
        if analysis_text:
            prompt_parts.append(analysis_text)
        
        prompt_parts.extend([
            "═══════════════════════════════════════════════════════════════════",
            "SOURCE CODE TO TEST",
            "═══════════════════════════════════════════════════════════════════",
            "```kotlin",
            source_code.strip()[:5000],
            "```",
        ])

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
            "□ NEVER pass null to non-nullable String parameter (check source signature first!)",
            "□ NEVER use Google Truth - use JUnit5 Assertions.* only",
            "□ If using TimeUnit → add: import java.util.concurrent.TimeUnit",
            "□ If using SimpleDateFormat → add: import java.text.SimpleDateFormat",
            "□ If using Locale → add: import java.util.Locale",
            "□ If using Date → add: import java.util.Date",
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

    def _validate_generated_tests(self, code: str, source_code: Optional[str] = None) -> List[str]:
        """
        Comprehensive validation of generated test code.
        Catches typos, syntax issues, common mistakes, AND logic errors.
        
        Args:
            code: Generated Kotlin test code
            source_code: Original source code (for contextual validation)
            
        Returns:
            List of validation error messages (empty if valid)
        """
        import re
        errors = []
        
        if not code or not code.strip():
            return ["Empty code generated"]
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTEXTUAL LOGIC VALIDATION (if source code provided)
        # ═══════════════════════════════════════════════════════════════════
        if source_code:
            analysis = analyze_kotlin_code(source_code)
            
            # Check for direct inner class instantiation
            for inner_class in analysis["inner_classes"]:
                # Pattern: ClassName(args) where ClassName is an inner class
                # But NOT: OuterClass().InnerClass() which is valid
                pattern = rf'(?<!\.)(?<!\w){inner_class}\s*\('
                if re.search(pattern, code):
                    # Make sure it's not being accessed through an outer instance
                    # Valid: adapter.ViewHolder(...) or ConversationsAdapter().ViewHolder(...)
                    # Invalid: ViewHolder(...) directly
                    outer_class = analysis["class_name"]
                    valid_pattern = rf'(?:{outer_class}\(\)|{outer_class.lower()}|adapter)\s*\.\s*{inner_class}'
                    if not re.search(valid_pattern, code, re.IGNORECASE):
                        errors.append(
                            f"LOGIC ERROR: '{inner_class}' is an inner class of '{outer_class}'. "
                            f"Cannot instantiate directly. Use '{outer_class.lower()}.{inner_class}()' "
                            f"or create through the outer class instance."
                        )
            
            # Check for private class access
            for private_class in analysis["private_classes"]:
                if re.search(rf'\b{private_class}\s*\(', code):
                    errors.append(
                        f"LOGIC ERROR: '{private_class}' is a private class. "
                        f"Cannot access or instantiate from test code. "
                        f"Test its behavior indirectly through public API."
                    )
            
            # Check for tests that don't set up adapter data before testing
            if analysis["class_type"] == "recycler_adapter":
                # If testing onBindViewHolder but not calling submitList
                if "onBindViewHolder" in code and "submitList" not in code:
                    # Check if there's any data setup
                    if "getItem(" in code or "currentList" in code:
                        errors.append(
                            "LOGIC ERROR: Testing adapter binding without data setup. "
                            "Call adapter.submitList(listOf(...)) before testing onBindViewHolder."
                        )
        
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
        # NULL SAFETY VALIDATION (Kotlin non-nullable types)
        # ═══════════════════════════════════════════════════════════════════

        # Detect tests passing null to functions
        null_calls = re.findall(r'\w+\(null\)', code)
        if null_calls:
            if source_code:
                # Check if the function signature accepts nullable (?)
                for call in null_calls:
                    func_name = call.split('(')[0]
                    # Look for function in source that takes non-nullable String
                    pattern = rf'fun\s+{func_name}\s*\(\s*\w+\s*:\s*String[^?]'
                    if re.search(pattern, source_code):
                        errors.append(
                            f"NULL SAFETY: '{call}' passes null to non-nullable String parameter. "
                            f"Remove this test or change parameter type to String? in source."
                        )
            else:
                errors.append(
                    f"NULL SAFETY: Detected null passed to function: {null_calls}. "
                    f"Verify the function accepts nullable type (String?) before using null."
                )

        # Detect Google Truth imports
        if "com.google.common.truth" in code:
            errors.append(
                "FORBIDDEN IMPORT: 'com.google.common.truth.Truth' (Google Truth) is not in dependencies. "
                "Use JUnit5 Assertions instead: assertEquals, assertNull, assertTrue, etc."
            )

        # Detect missing java.util imports
        java_util_needed = {
            "TimeUnit": "import java.util.concurrent.TimeUnit",
            "SimpleDateFormat": "import java.text.SimpleDateFormat",
            "Locale": "import java.util.Locale",
        }
        for keyword, needed_import in java_util_needed.items():
            if keyword in code and needed_import not in code:
                errors.append(
                    f"MISSING IMPORT: '{keyword}' used but '{needed_import}' not imported."
                )
        
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

LOGIC ERROR FIXES:
• INNER CLASS: If error says "inner class", you CANNOT instantiate it directly.
  - Remove tests that try to create inner class directly
  - Test through the outer class public API instead
  - Example: Instead of "ViewHolder(view)", test adapter behavior
• PRIVATE CLASS: If error says "private class", you cannot access it.
  - Remove tests for private classes
  - Test indirectly through public methods
• ADAPTER DATA: For RecyclerView adapters, call submitList() before testing bindings

REQUIRED FIXES:
1. Fix ALL typos in annotations and keywords
2. Balance all braces {{}}, parentheses (), brackets []
3. Use JUnit 5 imports: org.junit.jupiter.api.*
4. Remove any Android framework mocks (Context, View, ViewGroup)
5. Remove any private constant access (ClassName.PRIVATE_CONST)
6. Remove tests for inaccessible classes (inner, private)

FORBIDDEN:
• JUnit 4 imports (org.junit.Test, org.junit.Before)
• @RunWith, @Rule annotations
• Kotlin assert() function - use assertEquals, assertTrue, etc.
• Mocking Context, View, ViewGroup, LayoutInflater
• Direct instantiation of inner classes without outer instance
• Testing private nested classes

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
Return ONLY the corrected code in a single ```kotlin``` block.
Do NOT include explanations or comments outside the code block."""
        
        if source_code:
            # Include code analysis to help LLM understand what's testable
            analysis = analyze_kotlin_code(source_code)
            analysis_text = format_analysis_for_prompt(analysis)
            
            repair_prompt += f"""

═══════════════════════════════════════════════════════════════════
SOURCE CODE ANALYSIS (CONSTRAINTS TO FOLLOW)
═══════════════════════════════════════════════════════════════════
{analysis_text if analysis_text else "No special constraints detected."}

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
