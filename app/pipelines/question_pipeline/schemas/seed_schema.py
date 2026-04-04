from pydantic import BaseModel

from app.pipelines.question_pipeline.schemas.question_schema import DifficultyLevel, QuestionSplit


class QuestionSeed(BaseModel):
    """A deterministic seed from which one question is generated.

    The seed encodes everything except the natural language query text.
    The LLM receives the seed's context and difficulty to generate the query.
    The tool_call is computed before LLM invocation and never modified by it.

    Attributes:
        seed_id: Unique identifier for this seed (e.g., seed_0042).
        tool_name: Name of the correct tool to invoke (or 'no_action').
        tool_params: Ordered mapping of parameter names to their enum values.
        tool_call: Pre-built canonical tool call string, deterministic from tool_params.
        difficulty: Target difficulty tier instructing the LLM how to phrase the query.
        context_facts: Relevant skeleton facts provided as LLM generation context.
        doc_ids: Document IDs that contain the facts needed to answer this question.
        split: Dataset split assignment decided before generation.
        phrasing_index: Index distinguishing multiple seeds with the same tool_params
            but different difficulty or phrasing variant (0-based).
    """

    seed_id: str
    tool_name: str
    tool_params: dict[str, str | int]
    tool_call: str
    difficulty: DifficultyLevel
    context_facts: list[str]
    doc_ids: list[str]
    split: QuestionSplit
    phrasing_index: int = 0
