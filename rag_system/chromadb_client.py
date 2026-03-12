"""
ChromaDB Client - RAG System
"""

import chromadb
from chromadb.config import Settings
from loguru import logger
from typing import List, Dict, Any, Optional
import os

class ChromaDBClient:
    """Client pour gérer ChromaDB"""
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"✅ ChromaDB client initialized: {persist_directory}")
        
        self.tests_collection = None
        self.fixes_collection = None
        self.conventions_collection = None
        
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialise les collections ChromaDB"""
        
        try:
            self.tests_collection = self.client.get_or_create_collection(
                name="kotlin_tests",
                metadata={"description": "Tests unitaires Kotlin", "hnsw:space": "cosine"}
            )
            logger.info(f"✅ Collection 'kotlin_tests': {self.tests_collection.count()} documents")
        except Exception as e:
            logger.error(f"❌ Error creating tests collection: {e}")
        
        try:
            self.fixes_collection = self.client.get_or_create_collection(
                name="test_fixes",
                metadata={"description": "Corrections de tests", "hnsw:space": "cosine"}
            )
            logger.info(f"✅ Collection 'test_fixes': {self.fixes_collection.count()} documents")
        except Exception as e:
            logger.error(f"❌ Error creating fixes collection: {e}")
        
        try:
            self.conventions_collection = self.client.get_or_create_collection(
                name="project_conventions",
                metadata={"description": "Conventions projet", "hnsw:space": "cosine"}
            )
            logger.info(f"✅ Collection 'project_conventions': {self.conventions_collection.count()} documents")
        except Exception as e:
            logger.error(f"❌ Error creating conventions collection: {e}")
    
    def add_test(self, test_id: str, test_code: str, test_file: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            meta = metadata or {}
            meta.update({"file": test_file, "type": "kotlin_test"})
            
            self.tests_collection.add(
                documents=[test_code],
                ids=[test_id],
                metadatas=[meta]
            )
            logger.debug(f"✅ Test ajouté: {test_id}")
        except Exception as e:
            logger.error(f"❌ Error adding test {test_id}: {e}")
    
    def add_convention(self, convention_id: str, description: str, example: str, category: str):
        try:
            self.conventions_collection.add(
                documents=[f"{description}\n\nExample:\n{example}"],
                ids=[convention_id],
                metadatas={"category": category, "description": description, "type": "convention"}
            )
            logger.debug(f"✅ Convention ajoutée: {convention_id}")
        except Exception as e:
            logger.error(f"❌ Error adding convention {convention_id}: {e}")
    
    def search_similar_tests(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        try:
            results = self.tests_collection.query(query_texts=[query], n_results=n_results)
            
            similar_tests = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    similar_tests.append({
                        "code": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })
            
            logger.debug(f"🔍 Found {len(similar_tests)} similar tests")
            return similar_tests
        except Exception as e:
            logger.error(f"❌ Error searching similar tests: {e}")
            return []
    
    def search_conventions(self, query: str, category: Optional[str] = None, n_results: int = 3) -> List[Dict[str, Any]]:
        try:
            where = {"category": category} if category else None
            
            results = self.conventions_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            conventions = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    conventions.append({
                        "content": doc,
                        "category": results['metadatas'][0][i].get('category'),
                        "description": results['metadatas'][0][i].get('description'),
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })
            
            logger.debug(f"🔍 Found {len(conventions)} conventions")
            return conventions
        except Exception as e:
            logger.error(f"❌ Error searching conventions: {e}")
            return []

    def search_fixes(self, query: str, error_type: Optional[str] = None, n_results: int = 3) -> List[Dict[str, Any]]:
        try:
            where = {"error_type": error_type} if error_type else None

            results = self.fixes_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )

            fixes = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    fixes.append({
                        "fix_code": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })

            logger.debug(f"🔍 Found {len(fixes)} similar fixes")
            return fixes
        except Exception as e:
            logger.error(f"❌ Error searching fixes: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tests": self.tests_collection.count(),
            "fixes": self.fixes_collection.count(),
            "conventions": self.conventions_collection.count(),
            "total": self.tests_collection.count() + self.fixes_collection.count() + self.conventions_collection.count()
        }


_chromadb_client = None

def get_chromadb_client() -> ChromaDBClient:
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = ChromaDBClient()
    return _chromadb_client
