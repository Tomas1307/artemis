"""Chain method for LLM reasoning over retrieved chunks to determine tool calls."""

import re

from loguru import logger

from app.prompts.prompt_loader import prompt_loader


class LLMToolReasoner:
    """Reasons over retrieved document chunks to determine the correct tool call.

    Takes an operator query and top-K retrieved chunks, then uses chain-of-thought
    reasoning to analyze the evidence and produce a tool call in canonical format.

    Two modes of operation:
    - Direct: LLM generates the tool call directly from query + context.
    - Chain-of-thought (CoT): LLM first reasons about what evidence the chunks
      provide, which tool is appropriate, and what parameters match, then
      produces the final tool call.

    Attributes:
        _llm: LLM instance for generation.
    """

    def __init__(self, llm) -> None:
        """Initialize with an LLM provider.

        Args:
            llm: Any object with a generate(messages, **kwargs) -> str method.
        """
        self._llm = llm

    def reason_direct(self, query: str, chunks: list[dict], tools_json: str) -> str:
        """Generate a tool call directly from query and retrieved context.

        Args:
            query: Operator query from the control center.
            chunks: Top-K retrieved chunks, each with 'content' and 'doc_id'.
            tools_json: JSON string of the 10 MASA tool definitions.

        Returns:
            Tool call string in canonical format.
        """
        system_message = prompt_loader.get_system_message_by_type("tool_reason_direct")
        template = prompt_loader.get_prompt_template_by_type("tool_reason_direct")
        config = prompt_loader.get_config_by_type("tool_reason_direct")

        context = self._format_chunks(chunks)

        user_message = template.format(
            query=query,
            context=context,
            tools=tools_json,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        result = self._llm.generate(messages, **config)
        return self._extract_tool_call(result)

    def reason_cot(self, query: str, chunks: list[dict], tools_json: str) -> dict:
        """Generate a tool call using chain-of-thought reasoning.

        The LLM first analyzes the evidence in the chunks, identifies
        the relevant tool, determines parameter values, and then produces
        the final tool call. Returns both the reasoning and the call.

        Args:
            query: Operator query from the control center.
            chunks: Top-K retrieved chunks, each with 'content' and 'doc_id'.
            tools_json: JSON string of the 10 MASA tool definitions.

        Returns:
            Dict with 'reasoning' (full CoT text) and 'tool_call' (canonical format).
        """
        system_message = prompt_loader.get_system_message_by_type("tool_reason_cot")
        template = prompt_loader.get_prompt_template_by_type("tool_reason_cot")
        config = prompt_loader.get_config_by_type("tool_reason_cot")

        context = self._format_chunks(chunks)

        user_message = template.format(
            query=query,
            context=context,
            tools=tools_json,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        result = self._llm.generate(messages, **config)
        tool_call = self._extract_tool_call(result)

        return {
            "reasoning": result,
            "tool_call": tool_call,
        }

    def reason_batch(
        self,
        queries: list[str],
        chunks_per_query: list[list[dict]],
        tools_json: str,
        use_cot: bool = False,
    ) -> list[dict]:
        """Process multiple queries with their retrieved chunks.

        Args:
            queries: List of operator queries.
            chunks_per_query: List of chunk lists, one per query.
            tools_json: JSON string of tool definitions.
            use_cot: If True, use chain-of-thought reasoning.

        Returns:
            List of result dicts with 'tool_call' and optionally 'reasoning'.
        """
        results = []
        total = len(queries)

        for i, (query, chunks) in enumerate(zip(queries, chunks_per_query)):
            if (i + 1) % 25 == 0:
                logger.info(f"Reasoning over query {i + 1}/{total}")

            if use_cot:
                result = self.reason_cot(query, chunks, tools_json)
            else:
                result = {"tool_call": self.reason_direct(query, chunks, tools_json)}

            results.append(result)

        return results

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as context for the LLM prompt.

        Args:
            chunks: List of chunk dicts with doc_id, subtopic, and content.

        Returns:
            Formatted string with numbered chunks.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            doc_id = chunk.get("doc_id", "unknown")
            subtopic = chunk.get("subtopic", "")
            content = chunk.get("content", chunk.get("content_preview", ""))
            parts.append(f"[Chunk {i}] ({doc_id} > {subtopic})\n{content}")
        return "\n\n".join(parts)

    def _extract_tool_call(self, llm_output: str) -> str:
        """Extract the tool call from LLM output.

        Looks for canonical format patterns like:
        get_telemetry(module='jaguar',metric='temperature',timeframe_hours=4)

        Args:
            llm_output: Raw LLM output text.

        Returns:
            Extracted tool call string, or 'PARSE_ERROR' if not found.
        """
        tool_names = [
            "get_telemetry", "get_crew_status", "get_module_status",
            "send_alert", "send_message", "schedule_maintenance",
            "activate_protocol", "control_system", "calculate_trajectory",
            "request_supply", "no_action",
        ]

        for tool_name in tool_names:
            pattern = rf'{tool_name}\([^)]*\)'
            match = re.search(pattern, llm_output)
            if match:
                return match.group(0)

        if "no_action" in llm_output.lower():
            return "no_action"

        logger.warning(f"Could not extract tool call from: {llm_output[:200]}")
        return "PARSE_ERROR"
