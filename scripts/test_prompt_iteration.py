"""Prompt iteration test script for question generation.

Runs sample seeds through the LLM and prints results for quality evaluation.
Used to iteratively refine the question_generation.yaml prompt.

Usage:
    python scripts/test_prompt_iteration.py [--iteration N] [--all-tools]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llms.llm_nvidia import NvidiaLLM
from app.pipelines.question_pipeline.steps.step_01_generate_seeds import GenerateSeedsStep
from app.skeleton.skeleton_loader import skeleton_loader


def main(iteration: int = 1, all_tools: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"PROMPT ITERATION {iteration} — SAMPLE TEST")
    print(f"{'='*60}\n")

    skeleton = skeleton_loader.load()
    llm = NvidiaLLM()

    seed_step = GenerateSeedsStep(skeleton=skeleton, target_per_tool=200, random_seed=42)
    all_seeds = seed_step.execute()

    rng = random.Random(iteration * 100)

    if all_tools:
        tools_to_sample = [
            "get_telemetry",
            "get_crew_status",
            "get_module_status",
            "send_alert",
            "send_message",
            "schedule_maintenance",
            "activate_protocol",
            "control_system",
            "calculate_trajectory",
            "request_supply",
            "no_action",
        ]
        difficulties_per_tool = ["easy", "medium"]
    else:
        tools_to_sample = [
            "get_telemetry",
            "send_alert",
            "activate_protocol",
            "no_action",
            "calculate_trajectory",
        ]
        difficulties_per_tool = ["easy", "medium", "hard"]

    sample_seeds = []
    for tool in tools_to_sample:
        tool_seeds = [s for s in all_seeds if s.tool_name == tool]
        for diff in difficulties_per_tool:
            candidates = [s for s in tool_seeds if s.difficulty.value == diff]
            if candidates:
                sample_seeds.append(rng.choice(candidates))

    from app.prompts.prompt_loader import prompt_loader
    system_message = prompt_loader.get_system_message_by_type("query_generator")
    template = prompt_loader.get_prompt_template_by_type("query_generator")
    config = prompt_loader.get_config_by_type("query_generator")

    results = []
    for i, seed in enumerate(sample_seeds, 1):
        context_block = "\n".join(f"- {fact}" for fact in seed.context_facts) if seed.context_facts else "N/A"
        params_block = ", ".join(f"{k}={v!r}" for k, v in seed.tool_params.items()) if seed.tool_params else "none"

        prompt = template.format(
            tool_name=seed.tool_name,
            tool_call=seed.tool_call,
            difficulty=seed.difficulty.value,
            phrasing_index=seed.phrasing_index,
            params=params_block,
            context_facts=context_block,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        print(f"[{i:02d}/{len(sample_seeds)}] {seed.tool_name} | {seed.difficulty.value} | phrasing={seed.phrasing_index}")
        print(f"       tool_call: {seed.tool_call}")

        try:
            response = llm.generate(messages, **config)
            query = response.strip()
            word_count = len(query.split())
            print(f"       QUERY ({word_count}w): {query}")
        except Exception as exc:
            query = f"ERROR: {exc}"
            print(f"       ERROR: {exc}")

        print()
        results.append({
            "seed_id": seed.seed_id,
            "tool_name": seed.tool_name,
            "difficulty": seed.difficulty.value,
            "phrasing_index": seed.phrasing_index,
            "tool_call": seed.tool_call,
            "generated_query": query,
            "word_count": len(query.split()) if not query.startswith("ERROR") else 0,
        })

    output_path = Path(f"scripts/iteration_{iteration:02d}_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--all-tools", action="store_true", help="Test all 11 tools (2 difficulties each)")
    args = parser.parse_args()
    main(args.iteration, args.all_tools)
