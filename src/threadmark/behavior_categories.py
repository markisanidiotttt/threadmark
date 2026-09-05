OPERATION_TO_BEHAVIOR = {
    "membership_check": "membership_test",
    "non_membership_check": "membership_test",

    "continue": "skip_processing",
    "break": "stop_processing",
    "return": "exit_processing",
    "raise": "error_exit",

    "collection_add": "record_item",
    "collection_append": "record_item",
    "collection_extend": "record_item",

    "collection_update": "update_collection",

    "collection_remove": "remove_item",
    "collection_discard": "remove_item",
    "collection_pop": "remove_item",
    "collection_clear": "clear_collection",

    "periodic_modulo_check": "periodic_trigger",

    "equality_check": "predicate_check",
    "inequality_check": "predicate_check",
    "negated_predicate": "predicate_check",

    "boolean_and": "compound_condition",
    "boolean_or": "compound_condition",

    "assignment": "state_assignment",
    "function_call": "invoke_action",
    "condition": "generic_condition",
}


ALLOWED_BEHAVIORS = sorted(
    set(OPERATION_TO_BEHAVIOR.values())
)


def operation_to_behavior(operation: str) -> str | None:
    """Map a concrete AST operation to its abstract behavior category."""

    return OPERATION_TO_BEHAVIOR.get(operation)