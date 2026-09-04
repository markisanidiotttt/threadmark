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
    return (
        f"File: {chunk.file_path}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        f"{chunk.content}"
    )

    
def embed_chunks(
    chunks: list[CodeChunk],
    model: SentenceTransformer,
):
    texts = []
    
    for chunk in chunks:
        texts.append(chunk_to_text(chunk))
        
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )
    
    return embeddings


def search_chunks(
    query: str,
    chunks: list[CodeChunk],
    embeddings,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievalResult]:
    
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
    
    semantic_results = search_chunks(
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
    




# PYTHONPATH=src python -m threadmark.retrieval
from threadmark.repository import clone_repository
from threadmark.chunking import chunk_repository


if __name__ == "__main__":
    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    chunks = chunk_repository(repo_path)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = embed_chunks(chunks, model)

    query = "What happens when the Riot API returns HTTP status code 429?"

    results = search_chunks_hybrid(
        query,
        chunks,
        embeddings,
        model,
    )

    print(f"\nQuery: {query}\n")

    for result in results:
        chunk = result.chunk

        print(
            f"\n{result.score:.3f} | "
            f"{chunk.file_path}:"
            f"{chunk.start_line}-{chunk.end_line}"
        )

        print("-" * 60)
        print(chunk.content)