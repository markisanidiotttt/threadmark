from dataclasses import dataclass


EMBEDDING_MODELS = {
    "MiniLM": {
        "name": "all-MiniLM-L6-v2",
        "trust_remote_code": False,
    },
    "Jina Code": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "trust_remote_code": True,
    },
}

ACTIVE_EMBEDDING_MODEL = "MiniLM"


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
        query="How does the crawler restore its saved traversal progress?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=444,
        expected_end_line=458,
    ),

    EvalCase(
        query="What happens if the crawler was interrupted while processing a player?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=462,
        expected_end_line=479,
    ),

    EvalCase(
        query="How does the crawler avoid collecting duplicate match IDs?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=618,
        expected_end_line=628,
    ),
    
    EvalCase(
    query="What happens when the Riot API returns a temporary server error?",
    expected_file="src/data_collection/collect_matches.py",
    expected_start_line=62,
    expected_end_line=69,
),

    EvalCase(
        query="What happens when the Riot API key is invalid or expired?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=71,
        expected_end_line=74,
    ),

    EvalCase(
        query="How are previously collected matches loaded when collection resumes?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=227,
        expected_end_line=247,
    ),

    EvalCase(
        query="How does the crawler save its traversal state safely?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=316,
        expected_end_line=335,
    ),

    EvalCase(
        query="What happens to the current player if the API key expires during collection?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=559,
        expected_end_line=577,
    ),

    EvalCase(
        query="How does the crawler discover new players from downloaded matches?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=669,
        expected_end_line=690,
    ),

    EvalCase(
        query="What happens when a downloaded match is incomplete or invalid?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=697,
        expected_end_line=707,
    ),

    EvalCase(
        query="How does the crawler periodically checkpoint its progress?",
        expected_file="src/data_collection/collect_matches.py",
        expected_start_line=712,
        expected_end_line=725,
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

from sentence_transformers import SentenceTransformer, CrossEncoder

from threadmark.repository import clone_repository
from threadmark.chunking import (
    chunk_repository_fixed,
    chunk_repository_ast,
)
from threadmark.retrieval import (
    RERANKER_MODEL_NAME,
    embed_chunks,
    search_chunks_semantic,
    search_chunks_hybrid,
    search_chunks_bm25,
    rerank_results,
)
def evaluate_chunking(
    label: str,
    chunks,
    model,
    reranker,
):
    print(f"\n=== CHUNKING: {label} ===")

    embeddings = embed_chunks(chunks, model)

    summary = []

    for case in EVAL_CASES:
        semantic_results = search_chunks_semantic(
            case.query,
            chunks,
            embeddings,
            model,
            top_k=len(chunks),
        )

        bm25_results = search_chunks_bm25(
            case.query,
            chunks,
            top_k=len(chunks),
        )

        hybrid_results = search_chunks_hybrid(
            case.query,
            chunks,
            embeddings,
            model,
            top_k=len(chunks),
        )
        
        hybrid_candidates = hybrid_results[:20]
        
        reranked_results = rerank_results(
            case.query,
            hybrid_candidates,
            reranker,
            top_k=len(hybrid_candidates),
        )
        
        
            ###TEMP
        if case.query == "How does the crawler periodically checkpoint its progress?":
            print("\n=== RERANK DEBUG ===")

            for rank, result in enumerate(reranked_results[:10], start=1):
                chunk = result.chunk

                print(f"\n--- Rank {rank} | Score {result.score:.4f} ---")
                print(
                    f"{chunk.file_path}:"
                    f"{chunk.start_line}-{chunk.end_line}"
                )
                print(f"Symbol: {chunk.symbol_name}")
                print(chunk.content)
        
        
        
        reranked_rank = find_relevant_rank(
            reranked_results,
            case,
        )

        semantic_rank = find_relevant_rank(
            semantic_results,
            case,
        )

        bm25_rank = find_relevant_rank(
            bm25_results,
            case,
        )

        hybrid_rank = find_relevant_rank(
            hybrid_results,
            case,
        )

        summary.append(
            (
                case.query,
                semantic_rank,
                bm25_rank,
                hybrid_rank,
                reranked_rank,
            )
        )

    for (
        query, 
        semantic_rank, 
        bm25_rank, 
        hybrid_rank, 
        reranked_rank,
    ) in summary:
        print(f"\n{query}")
        print(f"  Semantic: Rank {semantic_rank}")
        print(f"  BM25:     Rank {bm25_rank}")
        print(f"  Hybrid:   Rank {hybrid_rank}")
        print(f"  Reranked: Rank {reranked_rank}")
            
if __name__ == "__main__":
    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )

    model_config = EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]

    model = SentenceTransformer(
        model_config["name"],
        trust_remote_code=model_config["trust_remote_code"],
    )

    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    fixed_chunks = chunk_repository_fixed(repo_path)
    ast_chunks = chunk_repository_ast(repo_path)

    evaluate_chunking(
        "Fixed Window",
        fixed_chunks,
        model,
        reranker,
    )

    evaluate_chunking(
        "AST Aware",
        ast_chunks,
        model,
        reranker,
    )