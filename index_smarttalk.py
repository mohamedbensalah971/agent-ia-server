from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

from config import settings
from rag_system.indexer import index_project

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 INDEXATION PROJET SMARTTALK")
    logger.info("=" * 60)
    
    # Chemin vers le projet SmartTalk
    project_path = settings.GIT_REPO_PATH
    
    if not project_path:
        logger.error("❌ GIT_REPO_PATH not set in .env")
        logger.info("   Please set: GIT_REPO_PATH=/path/to/SmartTalk-Android")
        exit(1)
    
    logger.info(f"📂 Project path: {project_path}")
    
    try:
        # Indexer le projet
        index_project(project_path)
        
        logger.info("=" * 60)
        logger.info("✅ INDEXATION TERMINÉE!")
        logger.info("=" * 60)
        
        # Afficher les stats
        from rag_system.chromadb_client import get_chromadb_client
        client = get_chromadb_client()
        stats = client.get_stats()
        
        logger.info(f"📊 Stats finales:")
        logger.info(f"   - Tests indexés: {stats['tests']}")
        logger.info(f"   - Corrections: {stats['fixes']}")
        logger.info(f"   - Conventions: {stats['conventions']}")
        logger.info(f"   - TOTAL: {stats['total']} documents")
        
    except Exception as e:
        logger.error(f"❌ Indexation failed: {e}")
        logger.exception("Full traceback:")
        exit(1)
