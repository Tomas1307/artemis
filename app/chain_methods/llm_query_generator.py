from loguru import logger

from app.llms.llm import BaseLLM
from app.prompts.prompt_loader import prompt_loader


class LLMQueryGenerator:
    """Chain method that generates MASA operator query text from a question seed.

    Takes a structured seed description (tool, params, difficulty, context facts)
    and produces a realistic natural language operator query. The correct tool call
    is built separately and never passed to this LLM.

    Args:
        llm: LLM provider instance for query generation.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._config = prompt_loader.get_config_by_type("query_generator")

    def generate_query(
        self,
        tool_name: str,
        tool_call: str,
        difficulty: str,
        phrasing_index: int,
        params: str,
        context_facts: str,
    ) -> str:
        """Generate a natural language operator query for a question seed.

        Args:
            tool_name: Name of the correct tool (used to guide style, not leaked).
            tool_call: Canonical tool call string (for internal reference only,
                must NOT appear in the generated query).
            difficulty: Difficulty tier — easy, medium, hard, or trap.
            phrasing_index: Zero-based phrasing variant index (ensures variation
                across multiple queries for the same tool call).
            params: Formatted parameter string for context.
            context_facts: Newline-separated skeleton facts relevant to this seed.

        Returns:
            Generated operator query string. Empty string if generation fails.
        """
        system_message = prompt_loader.get_system_message_by_type("query_generator")
        template = prompt_loader.get_prompt_template_by_type("query_generator")

        prompt = template.format(
            tool_name=tool_name,
            tool_call=tool_call,
            difficulty=difficulty,
            phrasing_index=phrasing_index,
            params=params,
            context_facts=context_facts,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.generate(messages, **self._config)
            return response.strip()
        except Exception as exc:
            logger.warning(f"Query generation error for {tool_name}/{difficulty}: {exc}")
            return ""
