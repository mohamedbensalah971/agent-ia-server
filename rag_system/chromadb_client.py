"""
ChromaDB Client - RAG System
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
from typing import List, Dict, Any, Optional
import os

from config import settings

class ChromaDBClient:
    """Client pour gérer ChromaDB"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        persist_directory = persist_directory or settings.RAG_CHROMA_PERSIST_DIR
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.embedding_function = self._build_embedding_function()
        
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
        
        _specs = [
            ("kotlin_tests",       "Tests unitaires Kotlin",  "tests_collection"),
            ("test_fixes",         "Corrections de tests",    "fixes_collection"),
            ("project_conventions","Conventions projet",      "conventions_collection"),
        ]
        
        for col_name, description, attr in _specs:
            setattr(self, attr, self._get_or_recreate_collection(col_name, description))

    def _get_or_recreate_collection(self, name: str, description: str):
        """Get or create a collection, deleting and recreating if the embedding function conflicts."""
        try:
            col = self.client.get_or_create_collection(
                name=name,
                metadata={"description": description, "hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            logger.info(f"✅ Collection '{name}': {col.count()} documents")
            return col
        except Exception as e:
            if "Embedding function conflict" in str(e):
                logger.warning(
                    f"⚠️ Embedding conflict on '{name}' (persisted vs new model). "
                    f"Deleting stale collection — please re-run index_smarttalk.py to re-index."
                )
                try:
                    self.client.delete_collection(name)
                    col = self.client.get_or_create_collection(
                        name=name,
                        metadata={"description": description, "hnsw:space": "cosine"},
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"✅ Collection '{name}' recreated fresh: {col.count()} documents")
                    return col
                except Exception as e2:
                    logger.error(f"❌ Error recreating collection '{name}': {e2}")
                    return None
            else:
                logger.error(f"❌ Error creating collection '{name}': {e}")
                return None

    def _build_embedding_function(self):
        """Initialise un modèle d'embeddings configurable pour améliorer la recherche code."""
        try:
            model_name = settings.RAG_EMBEDDING_MODEL
            device = settings.RAG_EMBEDDING_DEVICE
            logger.info(f"🧠 Loading embedding model: {model_name} (device={device})")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device
            )
        except Exception as e:
            # Fallback sûr: laisse ChromaDB utiliser son embedding par défaut.
            logger.warning(f"⚠️ Custom embeddings unavailable, fallback to default Chroma embeddings: {e}")
            return None

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure metadata only contains Chroma-compatible scalar values."""
        if not metadata:
            return {}

        clean: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                clean[str(key)] = value
            else:
                clean[str(key)] = str(value)

        return clean
    
    def add_test(self, test_id: str, test_code: str, test_file: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            meta = self._sanitize_metadata(metadata)
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
            metadata = self._sanitize_metadata({
                "category": category,
                "description": description,
                "type": "convention",
            })
            self.conventions_collection.add(
                documents=[f"{description}\n\nExample:\n{example}"],
                ids=[convention_id],
                metadatas=[metadata]
            )
            logger.debug(f"✅ Convention ajoutée: {convention_id}")
        except Exception as e:
            logger.error(f"❌ Error adding convention {convention_id}: {e}")
    
    def search_similar_tests(self, query: str, n_results: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            results = self.tests_collection.query(query_texts=[query], n_results=n_results, where=where)
            
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

    def _get_all_from_collection(self, collection, content_key: str) -> List[Dict[str, Any]]:
        """Fetch all documents from a collection for BM25 corpus building."""
        try:
            result = collection.get(include=["documents", "metadatas"])
            docs = []
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for i, doc in enumerate(documents):
                docs.append({
                    "id": ids[i] if i < len(ids) else str(i),
                    content_key: doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return docs
        except Exception as e:
            logger.error(f"❌ Error fetching all documents: {e}")
            return []

    def get_document_by_id(self, doc_id: str, collection_name: str = "kotlin_tests") -> Optional[str]:
        """Fetch a single document by ID — used for parent document retrieval."""
        collection_map = {
            "kotlin_tests": self.tests_collection,
            "test_fixes": self.fixes_collection,
            "project_conventions": self.conventions_collection,
        }
        collection = collection_map.get(collection_name)
        if collection is None:
            return None
        try:
            result = collection.get(ids=[doc_id], include=["documents"])
            docs = result.get("documents") or []
            return docs[0] if docs else None
        except Exception as e:
            logger.error(f"❌ Error fetching doc by id '{doc_id}': {e}")
            return None

    def get_all_tests(self) -> List[Dict[str, Any]]:
        return self._get_all_from_collection(self.tests_collection, content_key="code")

    def get_all_fixes(self) -> List[Dict[str, Any]]:
        return self._get_all_from_collection(self.fixes_collection, content_key="fix_code")

    def get_all_conventions(self) -> List[Dict[str, Any]]:
        return self._get_all_from_collection(self.conventions_collection, content_key="content")

    def store_approved_fix(
        self,
        fix_id: str,
        original_test_code: str,
        fix_code: str,
        error_type: Optional[str],
        error_message: str,
        test_file: str,
        confidence_score: float,
    ) -> bool:
        """Store an approved fix into test_fixes collection so it's retrievable for future errors."""
        try:
            document = (
                f"# Fix for {error_type or 'unknown'} error\n"
                f"# File: {test_file}\n"
                f"# Original error: {error_message[:200]}\n\n"
                f"# ORIGINAL TEST:\n{original_test_code[:800]}\n\n"
                f"# FIXED TEST:\n{fix_code}"
            )
            metadata = {
                "fix_id": fix_id,
                "error_type": error_type or "unknown",
                "error_message": error_message[:300],
                "test_file": test_file,
                "confidence_score": confidence_score,
                "approved": True,
                "source": "human_approved",
            }
            metadata = self._sanitize_metadata(metadata)
            self.fixes_collection.add(
                documents=[document],
                ids=[fix_id],
                metadatas=[metadata],
            )
            logger.info(f"✅ Approved fix stored in RAG: {fix_id} (error_type={error_type})")
            return True
        except Exception as e:
            logger.error(f"❌ Error storing approved fix {fix_id}: {e}")
            return False

    def update_fix_feedback(self, fix_id: str, approved: bool, feedback: Optional[str]) -> bool:
        """Record user feedback on a fix that was previously stored (upsert metadata)."""
        try:
            result = self.fixes_collection.get(ids=[fix_id], include=["documents", "metadatas"])
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            if not docs:
                logger.warning(f"⚠️ fix_id '{fix_id}' not found in fixes collection for feedback update")
                return False
            meta = dict(metas[0]) if metas else {}
            meta["approved"] = approved
            meta["user_feedback"] = (feedback or "")[:400]
            self.fixes_collection.update(ids=[fix_id], metadatas=[meta])
            logger.info(f"✅ Feedback recorded for fix {fix_id}: approved={approved}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating feedback for {fix_id}: {e}")
            return False


_chromadb_client = None

def get_chromadb_client() -> ChromaDBClient:
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = ChromaDBClient()
    return _chromadb_client
