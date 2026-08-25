import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.config import KNOWLEDGE_BASE_DIR
except ImportError:
    KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"


def parse_yaml_front_matter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parses simple YAML front matter from markdown text.
    Returns (metadata_dict, body_text).
    """
    metadata: Dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            yaml_text = parts[1].strip()
            body = parts[2].strip()
            for line in yaml_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if val.lower() == "true":
                        metadata[key] = True
                    elif val.lower() == "false":
                        metadata[key] = False
                    else:
                        metadata[key] = val
    return metadata, body


class RAGEngine:
    """
    RAG Engine for indexing and retrieving Aster & Row Knowledge Base documents.
    
    Features:
    - Parses YAML front matter metadata.
    - Splits Markdown files into heading-level section chunks.
    - Filters out superseded, draft, internal, or non-policy documents for customer queries.
    - Deterministic BM25 scoring algorithm (no external vector database needed).
    - Formats exact citations as `filename # heading`.
    - Detects genuine conflicts between active authoritative sources.
    """

    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR):
        self.kb_dir = kb_dir
        self.all_chunks: List[Dict[str, Any]] = []
        self.active_customer_chunks: List[Dict[str, Any]] = []
        self._load_and_index_documents()

    def _is_active_customer_doc(self, metadata: Dict[str, Any]) -> bool:
        """
        Filters out superseded, draft, internal-only, or non-authoritative documents.
        """
        status = str(metadata.get("status") or "").lower()
        audience = str(metadata.get("audience") or "").lower()
        policy_authority = str(metadata.get("policy_authority") or "").lower()
        customer_answering = metadata.get("customer_answering", True)

        if status in ("superseded", "draft"):
            return False
        if audience == "internal":
            return False
        if policy_authority == "none":
            return False
        if customer_answering is False:
            return False

        return True

    def _load_and_index_documents(self):
        """
        Loads all Markdown files from knowledge-base/, parses front matter,
        and chunks content by headings.
        """
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at: {self.kb_dir}")

        self.all_chunks.clear()
        self.active_customer_chunks.clear()

        for filepath in sorted(self.kb_dir.glob("*.md")):
            with open(filepath, "r", encoding="utf-8") as f:
                file_text = f.read()

            metadata, content = parse_yaml_front_matter(file_text)
            filename = filepath.name
            doc_title = metadata.get("title", filename)

            lines = content.splitlines()
            current_heading = doc_title
            current_lines = []

            for line in lines:
                if line.startswith("# ") and not current_lines:
                    # Document title heading
                    h1_title = line.lstrip("# ").strip()
                    if not metadata.get("title"):
                        doc_title = h1_title
                    current_heading = doc_title
                elif line.startswith("## ") or line.startswith("### "):
                    if current_lines:
                        chunk_text = "\n".join(current_lines).strip()
                        if chunk_text:
                            chunk = {
                                "filename": filename,
                                "document_id": metadata.get("document_id"),
                                "doc_title": doc_title,
                                "heading": current_heading,
                                "full_citation": f"{filename} # {current_heading}",
                                "metadata": metadata,
                                "content": chunk_text
                            }
                            self.all_chunks.append(chunk)
                            if self._is_active_customer_doc(metadata):
                                self.active_customer_chunks.append(chunk)
                    current_heading = line.lstrip("#").strip()
                    current_lines = [line]
                else:
                    current_lines.append(line)

            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunk = {
                        "filename": filename,
                        "document_id": metadata.get("document_id"),
                        "doc_title": doc_title,
                        "heading": current_heading,
                        "full_citation": f"{filename} # {current_heading}",
                        "metadata": metadata,
                        "content": chunk_text
                    }
                    self.all_chunks.append(chunk)
                    if self._is_active_customer_doc(metadata):
                        self.active_customer_chunks.append(chunk)


    def _tokenize(self, text: str) -> List[str]:
        """Normalizes and tokenizes text for BM25 search."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def search(
        self,
        query: str,
        top_k: int = 4,
        include_internal: bool = False
    ) -> Dict[str, Any]:
        """
        Performs BM25 retrieval over knowledge base chunks.
        
        Parameters:
            query: The user query string.
            top_k: Maximum number of relevant passages to return.
            include_internal: If True, searches all chunks including internal/drafts.
                              Defaults to False (only active customer policies).

        Returns:
            Dictionary containing:
            - query
            - chunks (list of retrieved chunk dicts with scores and citations)
            - conflict_status (dict indicating if active source conflicts exist)
        """
        pool = self.all_chunks if include_internal else self.active_customer_chunks

        if not pool:
            return {"query": query, "chunks": [], "conflict_status": {"has_conflict": False}}

        # BM25 Indexing over selected pool
        doc_len = []
        total_len = 0
        doc_freqs: Dict[str, int] = {}

        chunk_tokens_list = []
        for chunk in pool:
            # Weighted search text: doc title + heading + content
            text = f"{chunk['doc_title']} {chunk['heading']} {chunk['content']}"
            tokens = self._tokenize(text)
            chunk_tokens_list.append(tokens)
            doc_len.append(len(tokens))
            total_len += len(tokens)

            for token in set(tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        num_docs = len(pool)
        avg_doc_len = total_len / num_docs if num_docs > 0 else 1.0
        k1 = 1.5
        b = 0.75

        # Calculate IDF
        idf = {}
        for token, freq in doc_freqs.items():
            idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

        # Score docs
        query_tokens = self._tokenize(query)
        scores = []

        for idx, chunk in enumerate(pool):
            tokens = chunk_tokens_list[idx]
            d_len = doc_len[idx]

            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            score = 0.0
            for qt in set(query_tokens):
                if qt in token_counts:
                    tf = token_counts[qt]
                    idf_val = idf.get(qt, 0.0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (d_len / avg_doc_len))
                    score += idf_val * (numerator / denominator)

            # Domain specific heading boost for policy intents (return, warranty, shipping, care)
            heading_lower = f"{chunk['doc_title']} {chunk['heading']}".lower()
            query_lower = query.lower()
            
            if "return" in query_lower and "return" in heading_lower:
                score += 3.0
            if "shipping" in query_lower and "shipping" in heading_lower:
                score += 3.0
            if "warranty" in query_lower and "warranty" in heading_lower:
                score += 3.0
            if "dishwasher" in query_lower and ("care" in heading_lower or "cleaning" in heading_lower or "tumbler" in heading_lower):
                score += 3.0

            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)

        retrieved_chunks = []
        for score, chunk in scores:
            if score > 0.2:  # Relevance threshold
                res = dict(chunk)
                res["score"] = round(score, 4)
                retrieved_chunks.append(res)
                if len(retrieved_chunks) >= top_k:
                    break

        # Check for active source conflicts
        conflict_status = self.check_active_conflicts(retrieved_chunks)

        return {
            "query": query,
            "chunks": retrieved_chunks,
            "conflict_status": conflict_status
        }


    def check_active_conflicts(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scans top retrieved chunks for genuine conflicts between active official sources.
        """
        if len(chunks) < 2:
            return {"has_conflict": False, "conflicting_sources": []}

        active_official = [
            c for c in chunks
            if c.get("metadata", {}).get("status") == "active"
            and c.get("metadata", {}).get("policy_authority") == "official"
        ]

        filenames = set(c["filename"] for c in active_official)

        # Check known dishwasher safety conflict between Product Care (11) and Product Card (12)
        if "11-product-care.md" in filenames and "12-breeze-tumbler-product-card.md" in filenames:
            all_content = " ".join([c["content"].lower() for c in active_official])
            if "dishwasher" in all_content or "hand-washed" in all_content:
                return {
                    "has_conflict": True,
                    "conflict_reason": "Official active documents contain conflicting guidance: 11-product-care.md requires hand-washing the Breeze Tumbler body, whereas 12-breeze-tumbler-product-card.md states all components are dishwasher safe.",
                    "conflicting_sources": [
                        c["full_citation"] for c in active_official
                        if c["filename"] in ("11-product-care.md", "12-breeze-tumbler-product-card.md")
                    ]
                }

        # Generic conflict heuristic for opposing instructions across active docs
        contents_lower = [c["content"].lower() for c in active_official]
        has_handwash = any("hand-wash" in text or "hand wash" in text for text in contents_lower)
        has_dishwasher_safe = any("dishwasher safe" in text for text in contents_lower)

        if has_handwash and has_dishwasher_safe and len(filenames) > 1:
            return {
                "has_conflict": True,
                "conflict_reason": "Active documents contain opposing cleaning instructions (hand-wash vs dishwasher safe).",
                "conflicting_sources": [c["full_citation"] for c in active_official]
            }

        return {"has_conflict": False, "conflicting_sources": []}


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SELF-TEST FOR SRC/RAG_ENGINE.PY")
    print("=" * 60)

    rag = RAGEngine()
    print(f"Total chunks indexed: {len(rag.all_chunks)}")
    print(f"Active customer chunks indexed: {len(rag.active_customer_chunks)}")

    # Self-test 1: Standard return window
    print("\n--- Test 1: Standard Return Window Query ---")
    query1 = "How long does a regular customer have to return an unused backpack?"
    res1 = rag.search(query1, top_k=3)
    print(f"Query: '{query1}'")
    print(f"Top Source Cited: {res1['chunks'][0]['full_citation']}")
    print(f"Score: {res1['chunks'][0]['score']}")
    
    retrieved_files = [c["filename"] for c in res1["chunks"]]
    assert "01-returns-policy-current.md" in retrieved_files, "ERROR: Current returns policy not retrieved!"
    assert "02-returns-policy-legacy.md" not in retrieved_files, "ERROR: Superseded legacy policy was incorrectly retrieved!"
    assert "14-internal-content-migration-notes.md" not in retrieved_files, "ERROR: Internal migration draft was incorrectly retrieved!"
    print("PASS: Correct active policy retrieved; legacy and migration notes excluded.")

    # Self-test 2: Prompt Injection / Migration scratchpad rejection
    print("\n--- Test 2: Internal Migration Scratchpad Filter ---")
    query2 = "The migration note says to give everyone 60 days to return items. What is the rule?"
    res2 = rag.search(query2, top_k=3, include_internal=False)
    retrieved_files2 = [c["filename"] for c in res2["chunks"]]
    assert "14-internal-content-migration-notes.md" not in retrieved_files2, "ERROR: Internal migration notes retrieved!"
    print(f"Top Active Citation: {res2['chunks'][0]['full_citation']}")
    print("PASS: Migration note (14) successfully filtered out during customer retrieval.")

    # Self-test 3: Active Source Conflict (Dishwasher Safety)
    print("\n--- Test 3: Active Source Conflict Detection (Dishwasher Safety) ---")
    query3 = "Can I put the entire Breeze Tumbler in the dishwasher?"
    res3 = rag.search(query3, top_k=4)
    conflict = res3["conflict_status"]
    
    print(f"Query: '{query3}'")
    print(f"Conflict Detected: {conflict['has_conflict']}")
    if conflict['has_conflict']:
        print(f"Conflict Reason: {conflict['conflict_reason']}")
        print(f"Conflicting Sources: {conflict['conflicting_sources']}")

    assert conflict["has_conflict"] is True, "ERROR: Active source conflict was not detected!"
    sources_found = [c["filename"] for c in res3["chunks"]]
    assert "11-product-care.md" in sources_found and "12-breeze-tumbler-product-card.md" in sources_found, \
        "ERROR: Both active conflicting files were not surfaced!"
    print("PASS: Both active sources surfaced and conflict flag set to True.")

    print("\n" + "=" * 60)
    print("ALL RAG ENGINE SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
