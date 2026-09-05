import json
from dataclasses import dataclass
from threadmark.behavior_categories import ALLOWED_BEHAVIORS

from openai import OpenAI

QUERY_PLANNER_MODEL = "gpt-5.6-luna"

def load_frozen_query_plans(
    path: str = "eval/query_plans.json",
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@dataclass(frozen=True)
class PlannedRelation:
    source: str
    target: str


@dataclass
class QueryPlan:
    intent: str
    likely_behaviors: list[str]
    likely_relations: list[PlannedRelation]
    concepts: list[str]

    
def build_query_planner_prompt(query: str) -> str:
    """Build a prompt that converts a question into likely program behaviors."""

    behaviors = "\n".join(
        f"- {behavior}"
        for behavior in ALLOWED_BEHAVIORS
    )

    return f"""
You are planning a behavioral search over source code.

Given a natural-language question about a program, identify the abstract
program behaviors that code answering the question would likely exhibit.

Do not predict exact Python syntax, data structures, method names, variable
names, or implementation details.

These are behavioral search hypotheses, not claims about the repository.

Choose only behaviors from this vocabulary:

{behaviors}

Prefer a small number of strongly relevant behaviors. Do not include generic
behaviors unless they materially help identify the implementation.

Also infer a small number of directional relationships between behaviors.

A relationship source -> target means that the source behavior likely
controls, triggers, enables, or determines whether the target behavior occurs.

For example:
- checking whether an item was already seen may control skipping it
- checking whether an item is new may control recording it
- a periodic trigger may control a persistence action

Only propose relationships that are strongly implied by the question.
Both source and target must come from the allowed behavior vocabulary.

QUESTION:
{query}
""".strip()


def plan_query(query: str) -> QueryPlan:
    """Translate a natural-language query into likely implementation behaviors."""

    client = OpenAI()

    prompt = build_query_planner_prompt(query)

    response = client.responses.create(
        model=QUERY_PLANNER_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "query_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                        },
                        "likely_behaviors": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ALLOWED_BEHAVIORS,
                            },
                        },
                        "likely_relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "enum": ALLOWED_BEHAVIORS,
                                    },
                                    "target": {
                                        "type": "string",
                                        "enum": ALLOWED_BEHAVIORS,
                                    },
                                },
                                "required": [
                                    "source",
                                    "target",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "concepts": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "intent",
                        "likely_behaviors",
                        "likely_relations",
                        "concepts",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    )

    data = json.loads(response.output_text)
    
    relations = [
        PlannedRelation(
            source=relation["source"],
            target=relation["target"],
        )
        for relation in data["likely_relations"]
    ]

    return QueryPlan(
        intent=data["intent"],
        likely_behaviors=data["likely_behaviors"],
        likely_relations=relations,
        concepts=data["concepts"],
    )


def query_plan_from_dict(data: dict) -> QueryPlan:
    return QueryPlan(
        intent=data["intent"],
        likely_behaviors=data["likely_behaviors"],
        likely_relations=[
            PlannedRelation(
                source=relation["source"],
                target=relation["target"],
            )
            for relation in data["likely_relations"]
        ],
        concepts=data["concepts"],
    )


# PYTHONPATH=src python -m threadmark.query_planner

if __name__ == "__main__":
    queries = [
        "How does the crawler avoid collecting duplicate match IDs?",
        "How does the crawler discover new players from downloaded matches?",
        "How does the crawler periodically checkpoint its progress?",
    ]

    for query in queries:
        print(f"\nQUESTION: {query}")

        plan = plan_query(query)

        print(f"Intent: {plan.intent}")

        print("Behaviors:")
        for behaviors in plan.likely_behaviors:
            print(f"  - {behaviors}")
            
        print("Relations:")
        for relation in plan.likely_relations:
            print(
                f"  {relation.source}"
                f" -> "
                f"{relation.target}"
            )

        print("Concepts:")
        for concept in plan.concepts:
            print(f"  - {concept}")