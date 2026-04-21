"""
File Change Analyzer
Determines if code changes require test generation or if they're cosmetic.
"""

import re
from typing import Tuple, List
from loguru import logger


class FileChangeAnalyzer:
    """Analyze code changes to determine if tests are needed."""
    
    # Regex patterns for different change types
    COMMENT_PATTERN = re.compile(r'^\s*(//.*|/\*.*\*/)$', re.MULTILINE)
    WHITESPACE_PATTERN = re.compile(r'^\s*$', re.MULTILINE)
    DOCSTRING_PATTERN = re.compile(r'"""[\s\S]*?"""', re.MULTILINE)
    
    @staticmethod
    def is_cosmetic_change(original_code: str, modified_code: str) -> Tuple[bool, str]:
        """
        Determine if a change is cosmetic (only comments/whitespace).
        
        Args:
            original_code: Original source code
            modified_code: Modified source code
            
        Returns:
            Tuple of (is_cosmetic: bool, reason: str)
        """
        if not original_code or not modified_code:
            return False, "Cannot analyze empty code"
        
        # Remove comments and whitespace from both versions
        original_stripped = FileChangeAnalyzer._remove_cosmetic_elements(original_code)
        modified_stripped = FileChangeAnalyzer._remove_cosmetic_elements(modified_code)
        
        # If the functional code is identical, it's a cosmetic change
        if original_stripped == modified_stripped:
            changes = FileChangeAnalyzer._identify_cosmetic_changes(original_code, modified_code)
            reason = f"Only cosmetic changes: {', '.join(changes)}"
            return True, reason
        
        return False, "Functional code changes detected"
    
    @staticmethod
    def _remove_cosmetic_elements(code: str) -> str:
        """
        Remove comments, whitespace, and docstrings from code.
        
        Args:
            code: Source code
            
        Returns:
            Code with cosmetic elements removed
        """
        # Remove docstrings
        code = FileChangeAnalyzer.DOCSTRING_PATTERN.sub('', code)
        
        # Remove single-line comments (// ...)
        code = re.sub(r'\s*//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments (/* ... */)
        code = re.sub(r'\s*/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace (shrink multiple spaces/newlines)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'(\s*[{};,=()[\]])\s+', r'\1', code)
        
        return code.strip()
    
    @staticmethod
    def _identify_cosmetic_changes(original: str, modified: str) -> List[str]:
        """
        Identify what types of cosmetic changes were made.
        
        Args:
            original: Original code
            modified: Modified code
            
        Returns:
            List of change types identified
        """
        changes = []
        
        # Check for comment additions/changes
        original_comments = re.findall(r'//.*', original)
        modified_comments = re.findall(r'//.*', modified)
        if original_comments != modified_comments:
            changes.append("comments")
        
        # Check for whitespace changes (indentation, blank lines)
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        if len(original_lines) != len(modified_lines):
            changes.append("blank lines")
        
        # Check if any line content differs only in whitespace
        original_stripped_lines = [line.strip() for line in original_lines]
        modified_stripped_lines = [line.strip() for line in modified_lines]
        if original_stripped_lines != modified_stripped_lines:
            # If stripped lines are different, there's functional change
            return []
        else:
            changes.append("indentation")
        
        return changes or ["formatting"]
    
    @staticmethod
    def requires_test_generation(
        source_code: str,
        original_code: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if test generation is required.
        
        Args:
            source_code: The source code to evaluate
            original_code: Optional original code for comparison
            
        Returns:
            Tuple of (requires_tests: bool, reason: str)
        """
        # If no original code provided, assume tests are needed
        if original_code is None:
            return True, "No original code provided - assuming tests are needed"
        
        # Check if change is cosmetic
        is_cosmetic, reason = FileChangeAnalyzer.is_cosmetic_change(original_code, source_code)
        if is_cosmetic:
            logger.info(f"⏭️ Skipping test generation: {reason}")
            return False, reason
        
        # Check for actual functional code changes
        # Extract function/method definitions
        original_functions = FileChangeAnalyzer._extract_function_signatures(original_code)
        modified_functions = FileChangeAnalyzer._extract_function_signatures(source_code)
        
        if original_functions != modified_functions:
            return True, "Function signatures changed"
        
        # Extract class definitions
        original_classes = FileChangeAnalyzer._extract_class_signatures(original_code)
        modified_classes = FileChangeAnalyzer._extract_class_signatures(source_code)
        
        if original_classes != modified_classes:
            return True, "Class definitions changed"
        
        return True, "Potential functional changes detected"
    
    @staticmethod
    def _extract_function_signatures(code: str) -> List[str]:
        """Extract function signatures from code."""
        # Match: fun methodName(...) [: ReturnType] {
        pattern = r'\bfun\s+(\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{'
        return sorted(re.findall(pattern, code))
    
    @staticmethod
    def _extract_class_signatures(code: str) -> List[str]:
        """Extract class signatures from code."""
        # Match: class ClassName [: Parent] {
        pattern = r'\bclass\s+(\w+)(?:\s*:\s*[^{]+)?\s*\{'
        return sorted(re.findall(pattern, code))
    
    @staticmethod
    def analyze_code_quality(code: str) -> Tuple[List[str], List[str]]:
        """
        Analyze code for quality issues that might affect testing.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Tuple of (warnings, suggestions)
        """
        warnings = []
        suggestions = []
        
        # Check for missing error handling
        if 'try' not in code and 'catch' not in code and 'throw' not in code:
            suggestions.append("⚠️ No error handling found - consider adding try-catch blocks")
        
        # Check for side effects (logging, database calls, etc.)
        if re.search(r'\b(println|println|Log\.|database|write|save)\s*\(', code):
            suggestions.append("⚠️ Code contains side effects - ensure proper mocking in tests")
        
        # Check for complex logic
        if code.count('if') > 5 or code.count('for') > 3:
            suggestions.append("ℹ️ Complex control flow detected - consider comprehensive test coverage")
        
        # Check for null safety patterns
        if '!!' in code and '?.let' not in code:
            warnings.append("⚠️ Non-null assertion (!!), ) detected - ensure proper null handling in tests")
        
        return warnings, suggestions


# Optional import for type hints
from typing import Optional
