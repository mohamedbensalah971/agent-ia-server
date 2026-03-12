"""
Indexer - RAG System
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
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
        """Indexe un fichier de test complet"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Indexer le fichier entier
            test_id = f"{file_path.stem}"
            
            # Limiter la taille pour ChromaDB (max 1000 caractères)
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            
            self.chroma_client.add_test(
                test_id=test_id,
                test_code=content,
                test_file=str(file_path.relative_to(self.project_path)),
                metadata={
                    "file": file_path.name,
                    "type": "full_file"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in _index_test_file: {e}")
    
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