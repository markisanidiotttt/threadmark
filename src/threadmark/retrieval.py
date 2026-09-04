from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from threadmark.chunking import CodeChunk

MODEL_NAME = "all-MiniLM-L6-v2"

@dataclass
class RetrievalResult:
    chunk: CodeChunk
    score: float


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

    results = search_chunks(
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