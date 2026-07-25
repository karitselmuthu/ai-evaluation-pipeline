import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pydantic
from google.antigravity import Agent, LocalAgentConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger("RAG-Retriever")

class BM25Retriever:
    """
    Lightweight, clean BM25 retriever.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus  # Each item: {"text": str, "metadata": dict, "id": int}
        
        # Tokenize corpus
        self.doc_count = len(corpus)
        self.doc_lens = [len(self._tokenize(doc["text"])) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.doc_count)
        
        # Build frequency dictionaries
        self.doc_freqs = []
        self.word_df = {}
        
        for idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc["text"])
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            
            for token in set(tokens):
                self.word_df[token] = self.word_df.get(token, 0) + 1
                
        # Precompute IDFs
        self.word_idf = {}
        for word, df in self.word_df.items():
            # Standard BM25 IDF formula
            self.word_idf[word] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace & lowercasing tokenization
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Search corpus for query and return list of (doc_index, score) tuples.
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lens[idx]
            freq_dict = self.doc_freqs[idx]
            
            for token in query_tokens:
                if token in freq_dict:
                    tf = freq_dict[token]
                    idf = self.word_idf.get(token, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * (numerator / denominator)
            
            scores.append((idx, score))
            
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class RerankResponse(pydantic.BaseModel):
    """
    Schema for LLM-based reranking.
    """
    ranked_indices: List[int]
    explanations: List[str]


class RAGRetriever:
    """
    Handles Document chunking, Vector index building, BM25 building,
    and retrieving contexts under Config A, B, and C.
    """
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings_model = None
        self.chunk_embeddings: List[np.ndarray] = []
        self.bm25: BM25Retriever = None
        
        self._load_and_chunk_documents()
        self._build_indexes()

    @staticmethod
    def _extract_text(file_path: Path) -> str:
        """
        Returns the plain text of a document. `.txt`/`.md` are read as UTF-8;
        `.pdf` is parsed page-by-page with pypdf (page texts joined by blank
        lines so the paragraph splitter downstream treats page breaks as
        chunk boundaries). Unknown extensions fall back to a UTF-8 read.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise ImportError(
                    "Reading PDF documents requires 'pypdf'. Install it with "
                    "`pip install pypdf` (already listed in requirements.txt)."
                ) from e
            reader = PdfReader(str(file_path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n\n".join(pages)

        # .txt, .md, and any other text-like file
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_and_chunk_documents(self):
        """
        Loads all supported documents (.txt, .md, .pdf) in docs_dir and splits
        them into chunks.
        """
        if not self.docs_dir.exists():
            logger.warning(f"Documents directory {self.docs_dir} does not exist.")
            return

        text_files = (
            list(self.docs_dir.glob("**/*.txt"))
            + list(self.docs_dir.glob("**/*.md"))
            + list(self.docs_dir.glob("**/*.pdf"))
        )
        logger.info(f"Loading {len(text_files)} documents from {self.docs_dir}...")

        chunk_idx = 0
        for file_path in text_files:
            try:
                content = self._extract_text(file_path)
                if not content.strip():
                    logger.warning(f"No extractable text in {file_path.name}; skipping.")
                    continue

                # Clean up content
                content = content.replace("\r", "")
                
                # Split by double newlines or paragraph blocks
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                
                for p in paragraphs:
                    # If paragraph is very large, split into chunks of ~500 chars
                    if len(p) > 600:
                        words = p.split()
                        current_chunk = []
                        current_len = 0
                        for word in words:
                            current_chunk.append(word)
                            current_len += len(word) + 1
                            if current_len >= 500:
                                chunk_text = " ".join(current_chunk)
                                self.chunks.append({
                                    "id": chunk_idx,
                                    "text": chunk_text,
                                    "source": file_path.name
                                })
                                chunk_idx += 1
                                # Overlap: keep last 10 words
                                current_chunk = current_chunk[-10:]
                                current_len = sum(len(w) + 1 for w in current_chunk)
                        
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            self.chunks.append({
                                "id": chunk_idx,
                                "text": chunk_text,
                                "source": file_path.name
                            })
                            chunk_idx += 1
                    else:
                        self.chunks.append({
                            "id": chunk_idx,
                            "text": p,
                            "source": file_path.name
                        })
                        chunk_idx += 1
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")

        logger.info(f"Created {len(self.chunks)} text chunks.")

    def _build_indexes(self):
        """
        Builds Vector and BM25 indexes.
        """
        if not self.chunks:
            return

        # 1. Build BM25 Index
        self.bm25 = BM25Retriever(self.chunks)
        
        # 2. Build Embeddings
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No API key found in environment variables. Vector search might fail.")
            
        try:
            self.embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            
            texts = [c["text"] for c in self.chunks]
            # Embed in batches to prevent payload limits
            logger.info("Generating embeddings for document chunks...")
            batch_size = 50
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings_list.extend(self.embeddings_model.embed_documents(batch))
                
            self.chunk_embeddings = [np.array(emb) for emb in embeddings_list]
            logger.info("Embeddings generation complete.")
        except Exception as e:
            logger.error(f"Error creating Google GenAI embeddings: {e}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def retrieve_vector(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Config A: Vector-only retrieval
        """
        if not self.embeddings_model or not self.chunk_embeddings:
            logger.warning("Embeddings not built. Falling back to empty retrieval.")
            return []
            
        try:
            query_embedding = np.array(self.embeddings_model.embed_query(query))
            scores = []
            for idx, chunk_emb in enumerate(self.chunk_embeddings):
                sim = self._cosine_similarity(query_embedding, chunk_emb)
                scores.append((idx, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return [{**self.chunks[idx], "score": float(score)} for idx, score in scores[:top_n]]
        except Exception as e:
            logger.error(f"Vector retrieve failed: {e}")
            return []

    def retrieve_hybrid(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Config B: Hybrid (BM25 + Vector) Retrieval with Reciprocal Rank Fusion (RRF)
        """
        if not self.chunks:
            return []

        # Get Vector results
        vec_results = self.retrieve_vector(query, top_n=min(20, len(self.chunks)))
        # Get BM25 results
        bm25_tuples = self.bm25.search(query, top_n=min(20, len(self.chunks)))
        
        # Apply Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores = {}
        
        # Rank mapping helper
        for rank, item in enumerate(vec_results, 1):
            chunk_id = item["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            
        for rank, (idx, _) in enumerate(bm25_tuples, 1):
            chunk_id = self.chunks[idx]["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Sort chunk IDs by RRF score descending
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Retrieve original chunks in ranked order
        results = []
        for chunk_id, rrf_score in sorted_ids[:top_n]:
            chunk = next(c for c in self.chunks if c["id"] == chunk_id)
            results.append({**chunk, "score": rrf_score})
            
        return results

    async def retrieve_hybrid_rerank_async(self, query: str, top_n: int = 3, candidate_n: int = 10) -> List[Dict[str, Any]]:
        """
        Config C: Hybrid + LLM Reranker (Async)
        """
        # Fetch candidate list from Hybrid search
        candidates = self.retrieve_hybrid(query, top_n=min(candidate_n, len(self.chunks)))
        if not candidates:
            return []
            
        # Format the candidates into a prompt for Gemini
        candidate_text = ""
        for idx, cand in enumerate(candidates):
            candidate_text += f"[{idx}] (Source: {cand['source']})\n{cand['text']}\n---\n"
            
        prompt = f"""You are a highly precise information retrieval ranker. 
Your task is to rank the candidate passages in order of relevance to the query.
Rate each candidate passage on how well it directly answers or provides context for the question.

Question: {query}

Candidate Passages:
{candidate_text}

Provide the indices of the passages in order of relevance from most relevant to least relevant.
If a passage is completely irrelevant, do not include it. Provide brief 1-sentence reasons for your ranking decision for each selected passage.
"""
        config = LocalAgentConfig(
            model="gemini-3.5-flash",  # Fast and accurate for ranking tasks
            response_schema=RerankResponse
        )
        
        try:
            async with Agent(config=config) as agent:
                response = await agent.chat(prompt)
                result = await response.structured_output()
                
                if result and "ranked_indices" in result:
                    ranked_indices = result["ranked_indices"]
                    # Filter and reorder original candidates
                    reranked = []
                    for idx in ranked_indices:
                        if 0 <= idx < len(candidates):
                            reranked.append(candidates[idx])
                    
                    # Pad with remainder of candidates if reranker returned fewer than top_n
                    for cand in candidates:
                        if cand not in reranked:
                            reranked.append(cand)
                            
                    return reranked[:top_n]
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to standard hybrid results.")
            
        return candidates[:top_n]

    def retrieve_hybrid_rerank(self, query: str, top_n: int = 3, candidate_n: int = 10) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for hybrid rerank.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.retrieve_hybrid_rerank_async(query, top_n, candidate_n))
        else:
            return asyncio.run(self.retrieve_hybrid_rerank_async(query, top_n, candidate_n))

    def retrieve(self, query: str, config_type: str = "hybrid", top_n: int = 3) -> List[str]:
        """
        General retrieval method returning list of raw text contexts.
        config_type options: "vector", "hybrid", "hybrid_rerank".
        """
        if config_type == "vector":
            res = self.retrieve_vector(query, top_n=top_n)
        elif config_type == "hybrid":
            res = self.retrieve_hybrid(query, top_n=top_n)
        elif config_type == "hybrid_rerank":
            res = self.retrieve_hybrid_rerank(query, top_n=top_n)
        else:
            raise ValueError(f"Unknown retrieval config type: {config_type}")
            
        return [item["text"] for item in res]
