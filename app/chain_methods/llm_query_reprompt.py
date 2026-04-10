"""Chain method for generating multiple query variations for RRF retrieval."""

import re

from loguru import logger

from app.prompts.prompt_loader import prompt_loader


class LLMQueryReprompt:
    """Generates multiple reformulations of an operator query for RRF retrieval.

    Takes a raw operator query and produces 3 semantically diverse variations
    that capture different aspects of the information need. These variations
    are each sent through the bi-encoder independently, and the resulting
    rank lists are fused via RRF to improve retrieval quality.

    The llm argument accepts any object with a generate(messages) method,
    supporting both LocalLLM and NvidiaLLM interchangeably.

    Attributes:
        _llm: LLM instance for generation.
    """

    def __init__(self, llm) -> None:
        """Initialize with an LLM provider.

        Args:
            llm: Any object with a generate(messages, **kwargs) -> str method.
        """
        self._llm = llm

    def generate_variations(self, query: str, num_variations: int = 3) -> list[str]:
        """Generate multiple query reformulations for RRF retrieval.

        Args:
            query: Raw operator query from the control center.
            num_variations: Number of variations to generate.

        Returns:
            List of query variations (always includes the original query
            as the first element, followed by generated variations).
        """
        system_message = prompt_loader.get_system_message_by_type("query_reprompt")
        template = prompt_loader.get_prompt_template_by_type("query_reprompt")
        config = prompt_loader.get_config_by_type("query_reprompt")

        user_message = template.format(
            query=query,
            num_variations=num_variations,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        result = self._llm.generate(messages, **config)
        variations = self._parse_variations(result, num_variations)

        return [query] + variations

    def generate_variations_batch(
        self,
        queries: list[str],
        num_variations: int = 3,
    ) -> list[list[str]]:
        """Generate query variations for multiple queries.

        Args:
            queries: List of raw operator queries.
            num_variations: Number of variations per query.

        Returns:
            List of variation lists, one per query. Each inner list starts
            with the original query followed by generated variations.
        """
        results = []
        for i, query in enumerate(queries):
            if (i + 1) % 50 == 0:
                logger.info(f"Generating variations for query {i + 1}/{len(queries)}")
            variations = self.generate_variations(query, num_variations)
            results.append(variations)
        return results

    def _parse_variations(self, llm_output: str, expected_count: int) -> list[str]:
        """Parse numbered variations from LLM output.

        Handles formats like '1. ...', '1) ...', or '- ...'.

        Args:
            llm_output: Raw LLM output text.
            expected_count: Expected number of variations.

        Returns:
            List of parsed variation strings.
        """
        lines = llm_output.strip().split("\n")
        variations = []

        for line in lines:
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line.strip())
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 10:
                variations.append(cleaned)

        if len(variations) > expected_count:
            variations = variations[:expected_count]

        return variations
