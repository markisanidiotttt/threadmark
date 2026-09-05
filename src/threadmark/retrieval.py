import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from threadmark.chunking import CodeChunk

MODEL_NAME = "all-MiniLM-L6-v2"

@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def chunk_to_text(chunk: CodeChunk) -> str:
    """Converts list of chunks to one multi-line string"""
    parts = [
        f"File: {chunk.file_path}",
        f"Lines: {chunk.start_line}-{chunk.end_line}",
    ]

    if chunk.symbol_name is not None:
        parts.append(
            f"Symbol: {chunk.symbol_name}"
        )

    parts.append(chunk.content)

    return "\n".join(parts)

    
def embed_chunks(
    chunks: list[CodeChunk],
    model: SentenceTransformer,
):
    """Embeds vector values to chunks"""
    texts = []
    
    for chunk in chunks:
        texts.append(chunk_to_text(chunk))
        
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )
    
    return embeddings





def search_chunks_semantic(
    query: str,
    chunks: list[CodeChunk],
    embeddings,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rank code chunks by cosine similarity to the query embedding."""
    
    results = []
    
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    
    scores = embeddings @ query_embedding
    ranked_indices = scores.argsort()[::-1]
    top_indices = ranked_indices[:top_k]
    
    for index in top_indices:
        result = RetrievalResult(
            chunk=chunks[index],
            score=float(scores[index]),
        )
        
        results.append(result)
        
    return results


def search_chunks_bm25(
    query: str,
    chunks: list[CodeChunk],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rank code chunks by bm25 similarity to the query embedding."""
    
    corpus = [
        tokenize(chunk_to_text(chunk))
        for chunk in chunks
    ]
    
    bm25 = BM25Okapi(corpus)
    
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    ranked_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    
    for index in ranked_indices:
        results.append(
            RetrievalResult(
                chunk=chunks[index],
                score=float(scores[index]),
            )
        )
        
    return results


def search_chunks_hybrid(
    query: str,
    chunks: list[CodeChunk],
    embeddings,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Combine semantic and BM25 rankings using reciprocal rank fusion."""
    
    semantic_results = search_chunks_semantic(
        query,
        chunks,
        embeddings,
        model,
        top_k=len(chunks),
    )
    
    lexical_results = search_chunks_bm25(
        query,
        chunks,
        top_k=len(chunks),
    )
    
    rrf_scores = {}    
    k = 60
    
    for rank, result in enumerate(semantic_results, start=1):
        chunk = result.chunk
        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )
        
        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (k + rank)
        )
        
    for rank, result in enumerate(lexical_results, start=1):
        chunk = result.chunk
        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )
        
        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (k + rank)
        )
        
    chunk_lookup = {
        (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        ): chunk
        for chunk in chunks
    }
    
    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    
    results = []
    
    for key, score in ranked[:top_k]:
        results.append(
            RetrievalResult(
                chunk=chunk_lookup[key],
                score=score,
            )
        )
        
    return results
    


