import json

from threadmark.query_planner import plan_query


BENCHMARK_QUERIES = [
    "What happens when the Riot API returns HTTP status code 429?",
    "How does the crawler restore its saved traversal progress?",
    "What happens if the crawler was interrupted while processing a player?",
    "How does the crawler avoid collecting duplicate match IDs?",
    "What happens when the Riot API returns a temporary server error?",
    "What happens when the Riot API key is invalid or expired?",
    "How are previously collected matches loaded when collection resumes?",
    "How does the crawler save its traversal state safely?",
    "What happens to the current player if the API key expires during collection?",
    "How does the crawler discover new players from downloaded matches?",
    "What happens when a downloaded match is incomplete or invalid?",
    "How does the crawler periodically checkpoint its progress?",
]


def freeze_query_plans(
    output_path: str = "eval/query_plans.json",
) -> None:
    frozen_plans = {}

    for query in BENCHMARK_QUERIES:
        print(f"Planning: {query}")

        plan = plan_query(query)

        frozen_plans[query] = {
            "intent": plan.intent,
            "likely_behaviors": plan.likely_behaviors,
            "likely_relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                }
                for relation in plan.likely_relations
            ],
            "concepts": plan.concepts,
        }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            frozen_plans,
            file,
            indent=2,
        )


if __name__ == "__main__":
    freeze_query_plans()