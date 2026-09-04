from dataclasses import dataclass


@dataclass
class EvalCase:
    query: str
    expected_file: str
    expected_start_line: int
    expected_end_line: int
    
EVAL_CASES = [
    EvalCase(
        query="What happens when the Riot API returns HTTP status code 429?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=52,
        expected_end_line=60,
    ),

    EvalCase(
        query="How does the crawler restore its progress after being restarted?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=444,
        expected_end_line=479,
    ),

    EvalCase(
        query="How does the crawler avoid collecting duplicate match IDs?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=618,
        expected_end_line=628,
    ),
]


def chunk_matches_case(chunk, case: EvalCase) -> bool:
    if chunk.file_path != case.expected_file:
        return False
    
    return (
        chunk.start_line <= case.expected_end_line
        and chunk.end_line >= case.expected_start_line
    )
    
    
def find_relevant_rank(results, case: EvalCase) -> int | None:
    for rank, result in enumerate(results, start=1):
        if chunk_matches_case(result.chunk, case):
            return rank
        
    return None


# PYTHONPATH=src python -m threadmark.evaluation

from sentence_transformers import SentenceTransformer

from threadmark.repository import clone_repository
from threadmark.chunking import chunk_repository
from threadmark.retrieval import (
    MODEL_NAME,
    embed_chunks,
    search_chunks,
)

if __name__ == "__main__":
    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    chunks = chunk_repository(repo_path)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_chunks(chunks, model)
    summary = []

    for case in EVAL_CASES:
        all_results = search_chunks(
            case.query,
            chunks,
            embeddings,
            model,
            top_k=len(chunks),
        )

        full_rank = find_relevant_rank(all_results, case)

        summary.append(
            (
                case.query,
                full_rank,
            )
        )

    print("\n=== RETRIEVAL SUMMARY ===")

    for query, rank in summary:
        if rank is None:
            print(f"MISS | {query}")
        else:
            print(
                f"Rank {rank:>3} | "
                f"Top-5: {'HIT' if rank <= 5 else 'MISS'} | "
                f"{query}"
            )
