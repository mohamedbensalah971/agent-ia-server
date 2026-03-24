"""
Indexer - RAG System
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from rag_system.chromadb_client import get_chromadb_client


class ProjectIndexer:
    """Indexe les tests du projet SmartTalk"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.chroma_client = get_chromadb_client()
        
        if not self.project_path.exists():
            logger.warning(f"⚠️ Project path does not exist: {project_path}")
    
    def index_all_tests(self):
        """Indexe tous les tests du projet"""
        logger.info(f"📚 Indexing tests from: {self.project_path}")
        
        test_files = self._find_test_files()
        logger.info(f"   Found {len(test_files)} test files")
        
        indexed_count = 0
        for test_file in test_files:
            try:
                self._index_test_file(test_file)
                indexed_count += 1
            except Exception as e:
                logger.error(f"❌ Error indexing {test_file.name}: {e}")
        
        logger.info(f"✅ Indexed {indexed_count}/{len(test_files)} test files")
        
        stats = self.chroma_client.get_stats()
        logger.info(f"📊 ChromaDB stats: {stats}")
    
    def _find_test_files(self) -> List[Path]:
        """Trouve tous les fichiers de tests Kotlin"""
        test_files = []
        test_dir = self.project_path / "app" / "src" / "test"
        
        if test_dir.exists():
            for file in test_dir.rglob("*Test.kt"):
                test_files.append(file)
        
        return test_files
    
    def _index_test_file(self, file_path: Path):
        """Indexe un fichier de test et ses chunks Kotlin (class/fun)."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            relative_path = str(file_path.relative_to(self.project_path))
            doc_prefix = self._safe_id(relative_path)
            parent_id = f"{doc_prefix}__full"

            # Parent document: gardé pour récupération de contexte global.
            self.chroma_client.add_test(
                test_id=parent_id,
                test_code=content,
                test_file=relative_path,
                metadata={
                    "file": file_path.name,
                    "path": relative_path,
                    "type": "full_file",
                    "chunk_type": "full_file",
                    "parent_id": parent_id
                }
            )

            chunks = self._extract_kotlin_chunks(content)
            logger.debug(f"   {file_path.name}: {len(chunks)} chunks extracted")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_prefix}__chunk_{i:03d}"
                chunk_meta = {
                    "file": file_path.name,
                    "path": relative_path,
                    "type": "chunk",
                    "chunk_type": chunk["chunk_type"],
                    "function_name": chunk.get("function_name"),
                    "class_name": chunk.get("class_name"),
                    "is_test": chunk.get("is_test", False),
                    "parent_id": parent_id,
                    "chunk_index": i,
                }

                self.chroma_client.add_test(
                    test_id=chunk_id,
                    test_code=chunk["text"],
                    test_file=relative_path,
                    metadata=chunk_meta
                )
            
        except Exception as e:
            logger.error(f"❌ Error in _index_test_file: {e}")

    def _extract_kotlin_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Découpe un fichier Kotlin test en chunks utiles pour la recherche."""
        lines = content.splitlines()
        if not lines:
            return []

        chunks: List[Dict[str, Any]] = []

        class_match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        class_name: Optional[str] = class_match.group(1) if class_match else None

        function_starts: List[int] = []
        for idx, line in enumerate(lines):
            if re.match(r"^\s*fun\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", line):
                function_starts.append(idx)

        # Header chunk: package/imports/class signature for global context.
        if function_starts and function_starts[0] > 0:
            header_text = "\n".join(lines[:function_starts[0]]).strip()
            if header_text:
                chunks.append({
                    "chunk_type": "header",
                    "class_name": class_name,
                    "is_test": False,
                    "text": header_text[:2500],
                })

        for i, start in enumerate(function_starts):
            end = function_starts[i + 1] if i + 1 < len(function_starts) else len(lines)
            adjusted_start = self._move_start_to_annotations(lines, start)

            function_lines = lines[adjusted_start:end]
            function_text = "\n".join(function_lines).strip()
            if not function_text:
                continue

            name_match = re.search(r"\bfun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", function_text)
            function_name = name_match.group(1) if name_match else None
            is_test = "@Test" in function_text or bool(function_name and function_name.lower().startswith("test"))

            chunks.append({
                "chunk_type": "test_method" if is_test else "method",
                "function_name": function_name,
                "class_name": class_name,
                "is_test": is_test,
                "text": function_text[:3000],
            })

        # Fallback: aucun fun détecté, indexer une version réduite du fichier.
        if not chunks:
            chunks.append({
                "chunk_type": "raw",
                "class_name": class_name,
                "is_test": "@Test" in content,
                "text": content[:3000],
            })

        return chunks

    def _move_start_to_annotations(self, lines: List[str], function_start: int) -> int:
        """Inclut les annotations Kotlin au-dessus de `fun` dans le chunk."""
        i = function_start
        while i > 0 and lines[i - 1].strip().startswith("@"):
            i -= 1
        return i

    def _safe_id(self, value: str) -> str:
        """Convertit un path en id stable compatible ChromaDB."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    
    def index_conventions(self):
        """Indexe les conventions du projet SmartTalk"""
        logger.info("📖 Indexing project conventions...")
        
        conventions = [
            {
                "id": "conv_junit5",
                "description": "Use JUnit 5 for testing framework",
                "example": "@Test\nfun testSomething() { ... }",
                "category": "framework"
            },
            {
                "id": "conv_mockk",
                "description": "Use MockK for mocking dependencies",
                "example": "@MockK\nprivate lateinit var apiService: ApiService\n\n@Before\nfun setup() {\n    MockKAnnotations.init(this)\n}",
                "category": "mocking"
            },
            {
                "id": "conv_koin",
                "description": "Use Koin for dependency injection in tests",
                "example": "startKoin {\n    modules(testModule)\n}",
                "category": "dependency_injection"
            },
            {
                "id": "conv_coroutines",
                "description": "Use TestCoroutineDispatcher for testing coroutines",
                "example": "@get:Rule\nval coroutineRule = TestCoroutineRule()",
                "category": "coroutines"
            },
            {
                "id": "conv_assertions",
                "description": "Use Truth assertions library",
                "example": "assertThat(result).isEqualTo(expected)",
                "category": "assertions"
            },
            {
                "id": "conv_given_when_then",
                "description": "Structure tests with Given-When-Then pattern",
                "example": "// Given\nval input = ...\n// When\nval result = ...\n// Then\nassertThat(result)...",
                "category": "structure"
            }
        ]
        
        for conv in conventions:
            self.chroma_client.add_convention(
                convention_id=conv["id"],
                description=conv["description"],
                example=conv["example"],
                category=conv["category"]
            )
        
        logger.info(f"✅ Indexed {len(conventions)} conventions")


def index_project(project_path: str):
    """
    Fonction helper pour indexer un projet
    
    Args:
        project_path: Chemin vers le projet SmartTalk-Android
    """
    indexer = ProjectIndexer(project_path)
    
    # Indexer les conventions
    indexer.index_conventions()
    
    # Indexer les tests existants
    indexer.index_all_tests()
    
    logger.info("🎉 Project indexing complete!")