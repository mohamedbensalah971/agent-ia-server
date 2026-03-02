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
