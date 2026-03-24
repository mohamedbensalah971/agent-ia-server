"""
Retriever - RAG System
"""

from typing import Dict, Any, List, Optional
import json
import re

from groq import Groq
from loguru import logger
from config import settings
from rag_system.chromadb_client import get_chromadb_client


class BM25Index:
    """Keyword BM25 index for hybrid retrieval alongside vector similarity."""

    _SPLIT_RE = re.compile(r"[^a-zA-Z0-9@_.]+")

    def __init__(self, documents: List[Dict[str, Any]], content_key: str):
        self.documents = documents
        self.content_key = content_key
        tokenized = [self._tokenize(d.get(content_key) or "") for d in documents]
        self.bm25 = None
        if tokenized:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi(tokenized)
            except ImportError:
                logger.warning("⚠️ rank_bm25 not installed – BM25 disabled. pip install rank-bm25")

    def _tokenize(self, text: str) -> List[str]:
        tokens = self._SPLIT_RE.split(text.lower())
        return [t for t in tokens if len(t) > 1]

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        results = []
        for i in top_idx:
            if scores[i] > 0.0:
                result = dict(self.documents[i])
                result["bm25_score"] = float(scores[i])
                results.append(result)
        return results


class RAGRetriever:
    def __init__(self):
        self.chroma_client = get_chromadb_client()
        self.llm_client = None
        self.llm_model = settings.GROQ_MODEL
        try:
            self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            logger.warning(f"⚠️ Groq client unavailable in retriever, using fallback query expansion: {e}")
        self._bm25_indices: Dict[str, tuple] = {}
    
    def get_context_for_fix(self, test_code: str, error_logs: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"🔍 Retrieving RAG context for fix (error_type={error_type})")

        # Base queries from live inputs.
        test_query = test_code[:500]
        error_query = error_logs[:300]

        # Phase 4: Multi-query + HyDE query transformation.
        expanded_error_queries = self._generate_multi_queries(error_query, error_type=error_type, max_queries=4)
        hyde_query = self._generate_hyde_query(test_code=test_code, error_logs=error_logs, error_type=error_type)

        # Keep ordering stable: base first, then expansions, then HyDE.
        all_error_queries = self._dedupe_keep_order([error_query, *expanded_error_queries, hyde_query])
        all_test_queries = self._dedupe_keep_order([test_query, hyde_query, *expanded_error_queries])

        tests_batches: List[List[Dict[str, Any]]] = []
        fixes_batches: List[List[Dict[str, Any]]] = []
        conventions_batches: List[List[Dict[str, Any]]] = []

        for query in all_test_queries[:5]:
            # Phase 6c: Prefer test_method chunks; fall back to unfiltered if none found.
            results = self.chroma_client.search_similar_tests(
                query=query, n_results=3, where={"chunk_type": "test_method"}
            )
            if not results:
                results = self.chroma_client.search_similar_tests(query=query, n_results=3)
            tests_batches.append(results)

        for query in all_error_queries[:5]:
            fixes_batches.append(self.chroma_client.search_fixes(query=query, error_type=error_type, n_results=3))
            conventions_batches.append(self.chroma_client.search_conventions(query=query, n_results=4))

        # Phase 5: BM25 hybrid search — keyword-exact matching in parallel with semantic.
        bm25_tests = self._get_bm25_index("tests")
        bm25_fixes = self._get_bm25_index("fixes")
        bm25_conventions = self._get_bm25_index("conventions")

        if bm25_tests:
            for query in all_test_queries[:3]:
                batch = bm25_tests.search(query, n_results=4)
                if batch:
                    tests_batches.append(batch)

        if bm25_fixes:
            for query in all_error_queries[:3]:
                batch = bm25_fixes.search(query, n_results=4)
                if batch:
                    fixes_batches.append(batch)

        if bm25_conventions:
            for query in all_error_queries[:3]:
                batch = bm25_conventions.search(query, n_results=4)
                if batch:
                    conventions_batches.append(batch)

        similar_tests = self._fuse_ranked_results(tests_batches, content_key="code", max_items=8)
        similar_fixes = self._fuse_ranked_results(fixes_batches, content_key="fix_code", max_items=8)
        conventions = self._fuse_ranked_results(conventions_batches, content_key="content", max_items=8)

        # Phase 5b: LLM re-ranking — score and filter top candidates by relevance.
        rerank_query = f"{error_type or ''} {error_query}".strip()
        similar_tests = self._rerank_candidates(similar_tests, rerank_query, content_key="code", top_k=4)
        similar_fixes = self._rerank_candidates(similar_fixes, rerank_query, content_key="fix_code", top_k=4)
        conventions = self._rerank_candidates(conventions, rerank_query, content_key="content", top_k=4)

        # Phase 6a: Prepend parent-file imports to function-level test chunks.
        similar_tests = self._enrich_with_parent_imports(similar_tests)

        # Phase 6b: Batch-compress chunks to keep only query-relevant lines.
        similar_tests = self._compress_candidates(similar_tests, rerank_query, content_key="code")
        similar_fixes = self._compress_candidates(similar_fixes, rerank_query, content_key="fix_code")

        logger.info(
            "   queries={} | similar_tests={}, fixes={}, conventions={} (phase6)",
            len(all_error_queries),
            len(similar_tests),
            len(similar_fixes),
            len(conventions),
        )

        return {
            "similar_tests": similar_tests,
            "similar_fixes": similar_fixes,
            "conventions": conventions,
            "error_type": error_type,
            "rag_queries": all_error_queries[:5],
            "hyde_query": hyde_query,
        }

    def warmup(self) -> None:
        """Preload local retrieval components so the first live request avoids cold-start latency."""
        try:
            self._get_bm25_index("tests")
            self._get_bm25_index("fixes")
            self._get_bm25_index("conventions")

            # Trigger one lightweight embedding pass per populated collection.
            if self.chroma_client.conventions_collection and self.chroma_client.conventions_collection.count() > 0:
                self.chroma_client.search_conventions("mockk", n_results=1)
            if self.chroma_client.tests_collection and self.chroma_client.tests_collection.count() > 0:
                self.chroma_client.search_similar_tests("kotlin test", n_results=1)

            logger.info("✅ RAG retriever warmed up")
        except Exception as e:
            logger.warning(f"⚠️ RAG warmup skipped: {e}")

    def _get_bm25_index(self, collection_name: str) -> Optional["BM25Index"]:
        """Lazily build and cache a BM25 index per collection; rebuild when doc count changes."""
        if collection_name == "tests":
            docs = self.chroma_client.get_all_tests()
            content_key = "code"
        elif collection_name == "fixes":
            docs = self.chroma_client.get_all_fixes()
            content_key = "fix_code"
        elif collection_name == "conventions":
            docs = self.chroma_client.get_all_conventions()
            content_key = "content"
        else:
            return None

        if not docs:
            return None

        doc_count = len(docs)
        cached = self._bm25_indices.get(collection_name)
        if cached and cached[0] == doc_count:
            return cached[1]

        index = BM25Index(docs, content_key)
        self._bm25_indices[collection_name] = (doc_count, index)
        logger.debug(f"🔑 BM25 index rebuilt for '{collection_name}': {doc_count} docs")
        return index

    def _rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        content_key: str,
        top_k: int = 4,
        min_score: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """LLM re-ranking: score candidates for relevance, drop weak results."""
        if not candidates or len(candidates) <= 2 or not self.llm_client:
            return candidates[:top_k]

        snippets = [c.get(content_key, "")[:300] for c in candidates[:8]]
        snippets_text = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets))

        prompt = (
            f"Rate each snippet's relevance (0-10) for fixing a Kotlin test failure.\n\n"
            f"ERROR CONTEXT:\n{query[:400]}\n\n"
            f"SNIPPETS:\n{snippets_text}\n\n"
            f'Return ONLY valid JSON: {{"scores": [5, 8, 3, ...]}}\n'
            f"One integer per snippet in the same order."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            payload = self._parse_json_object(response.choices[0].message.content or "")
            scores = payload.get("scores", [])
            if isinstance(scores, list) and len(scores) == len(snippets):
                scored = [
                    {**dict(candidates[i]), "rerank_score": float(scores[i])}
                    for i in range(len(snippets))
                ]
                filtered = [c for c in scored if c["rerank_score"] >= min_score]
                reranked = sorted(filtered, key=lambda x: x["rerank_score"], reverse=True)
                return reranked[:top_k] if reranked else candidates[:top_k]
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed, using original order: {e}")

        return candidates[:top_k]

    def _extract_imports(self, kotlin_code: str) -> str:
        """Extract package and import declarations from a Kotlin file."""
        lines = [
            line for line in kotlin_code.splitlines()
            if line.strip().startswith("package ") or line.strip().startswith("import ")
        ]
        return "\n".join(lines)

    def _enrich_with_parent_imports(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """For function-level chunks, prepend parent file's imports for full import context."""
        enriched: List[Dict[str, Any]] = []
        for candidate in candidates:
            meta = candidate.get("metadata") or {}
            chunk_type = meta.get("chunk_type", "")
            parent_id = meta.get("parent_id")
            # Only enrich non-full-file chunks that carry a parent reference.
            if chunk_type in ("test_method", "method", "header") and parent_id and chunk_type != "full_file":
                parent_text = self.chroma_client.get_document_by_id(parent_id, collection_name="kotlin_tests")
                if parent_text:
                    imports = self._extract_imports(parent_text)
                    if imports:
                        enriched_candidate = dict(candidate)
                        enriched_candidate["code"] = (
                            f"// [context: imports from parent file]\n{imports}\n\n{candidate.get('code', '')}"
                        )
                        enriched.append(enriched_candidate)
                        continue
            enriched.append(candidate)
        return enriched

    def _compress_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        content_key: str,
    ) -> List[Dict[str, Any]]:
        """Single batch LLM call: extract only query-relevant lines from each chunk."""
        if not candidates or not self.llm_client:
            return candidates

        # Only compress chunks that are long enough to benefit.
        to_compress = [(i, c) for i, c in enumerate(candidates) if len(c.get(content_key) or "") > 200]
        if not to_compress:
            return candidates

        snippets = [(c.get(content_key) or "")[:600] for _, c in to_compress]
        snippets_text = "\n\n".join(f"[{j + 1}] {s}" for j, s in enumerate(snippets))

        prompt = (
            "For each snippet below, extract only the lines relevant to fixing this error. "
            "Remove unrelated test methods. Keep all imports, annotations, and relevant APIs.\n\n"
            f"ERROR CONTEXT:\n{query[:400]}\n\n"
            f"SNIPPETS:\n{snippets_text}\n\n"
            'Return ONLY valid JSON: {"results": ["compressed snippet 1", ...]}\n'
            "One string per snippet in the same order."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=900,
            )
            payload = self._parse_json_object(response.choices[0].message.content or "")
            results = payload.get("results", [])
            if isinstance(results, list) and len(results) == len(to_compress):
                updated = list(candidates)
                for j, (original_idx, _) in enumerate(to_compress):
                    compressed = results[j].strip() if isinstance(results[j], str) else ""
                    if compressed:
                        updated[original_idx] = dict(candidates[original_idx])
                        updated[original_idx]["compressed_text"] = compressed
                return updated
        except Exception as e:
            logger.warning(f"⚠️ Batch compression failed, using original chunks: {e}")

        return candidates

    def _generate_multi_queries(self, base_query: str, error_type: Optional[str], max_queries: int = 4) -> List[str]:
        """Generate semantically different search queries for recall boost."""
        if not base_query.strip():
            return []

        if self.llm_client is None:
            return self._fallback_multi_queries(base_query, error_type, max_queries)

        prompt = f"""
You are helping with retrieval for Kotlin/Android test failures.
Given the error context, generate {max_queries} SHORT alternative search queries.

Constraints:
- Keep each query <= 140 characters.
- Focus on technical retrieval terms (framework, api, annotation, stack trace keywords).
- Do not repeat the original wording.
- Return strict JSON only in this format:
{{"queries": ["q1", "q2"]}}

error_type: {error_type or "unknown"}
error_context:
{base_query[:700]}
""".strip()

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=220,
            )
            content = (response.choices[0].message.content or "").strip()
            payload = self._parse_json_object(content)
            queries = payload.get("queries", []) if isinstance(payload, dict) else []
            cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            return cleaned[:max_queries]
        except Exception as e:
            logger.warning(f"⚠️ Multi-query generation failed, using heuristic fallback: {e}")
            return self._fallback_multi_queries(base_query, error_type, max_queries)

    def _generate_hyde_query(self, test_code: str, error_logs: str, error_type: Optional[str]) -> str:
        """Generate a hypothetical fix snippet (HyDE) and use it as search query."""
        if self.llm_client is None:
            return f"{error_type or 'unknown'} {error_logs[:220]} {test_code[:220]}"

        prompt = f"""
Produce a compact hypothetical Kotlin test fix snippet for retrieval purposes.

Rules:
- Return plain text only (no markdown).
- 6 to 12 lines max.
- Include realistic APIs/annotations likely needed to fix the issue.
- Prefer concrete symbols over explanations.

error_type: {error_type or "unknown"}
error_logs:
{error_logs[:650]}

failing_test_excerpt:
{test_code[:650]}
""".strip()

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=260,
            )
            hyde = (response.choices[0].message.content or "").strip()
            if hyde:
                return hyde[:900]
        except Exception as e:
            logger.warning(f"⚠️ HyDE generation failed, using deterministic fallback: {e}")

        return f"{error_type or 'unknown'} {error_logs[:220]} {test_code[:220]}"

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        """Parse JSON object from raw model output (supports fenced and mixed outputs)."""
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = candidate.replace("json", "", 1).strip()

        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(candidate[start:end + 1])
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
        return {}

    def _fallback_multi_queries(self, base_query: str, error_type: Optional[str], max_queries: int) -> List[str]:
        """Heuristic alternatives when LLM query expansion is unavailable."""
        normalized = " ".join(base_query.split())
        prefixes = [
            "kotlin junit mockk failure",
            "android test fix",
            "stacktrace root cause",
            "unit test dependency injection",
        ]
        candidates = [f"{p} {error_type or ''} {normalized[:100]}".strip() for p in prefixes]
        return candidates[:max_queries]

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _fuse_ranked_results(
        self,
        batches: List[List[Dict[str, Any]]],
        content_key: str,
        max_items: int = 6,
    ) -> List[Dict[str, Any]]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        if not batches:
            return []

        by_content: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}

        for batch in batches:
            for rank, item in enumerate(batch, start=1):
                content = (item.get(content_key) or "").strip()
                if not content:
                    continue

                # RRF with k=60 (standard robust default).
                scores[content] = scores.get(content, 0.0) + 1.0 / (60.0 + rank)

                if content not in by_content:
                    by_content[content] = item
                else:
                    existing_distance = by_content[content].get("distance")
                    new_distance = item.get("distance")
                    if new_distance is not None and (existing_distance is None or new_distance < existing_distance):
                        by_content[content] = item

        ranked_contents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        merged: List[Dict[str, Any]] = []
        for content, fused_score in ranked_contents[:max_items]:
            result = dict(by_content[content])
            result["fused_score"] = round(fused_score, 6)
            merged.append(result)

        return merged
    
    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        formatted = []

        if context.get("conventions"):
            formatted.append("=== PROJECT CONVENTIONS ===")
            for conv in context["conventions"][:3]:
                formatted.append(f"- {conv.get('description', conv.get('content', ''))}")

        if context.get("similar_tests"):
            formatted.append("\n=== SIMILAR TESTS FROM PROJECT ===")
            for test in context["similar_tests"][:2]:
                # Phase 6b: prefer compressed version if available.
                text = test.get("compressed_text") or test.get("code", "")
                formatted.append(text[:500])

        if context.get("similar_fixes"):
            formatted.append("\n=== KNOWN FIXES FOR THIS ERROR TYPE ===")
            for fix in context["similar_fixes"][:2]:
                text = fix.get("compressed_text") or fix.get("fix_code", "")
                formatted.append(text[:500])

        return "\n".join(formatted)


_rag_retriever = None

def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever
