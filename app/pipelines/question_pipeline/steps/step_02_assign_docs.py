from loguru import logger

from app.pipelines.question_pipeline.schemas.seed_schema import QuestionSeed
from app.pipelines.question_pipeline.utils.doc_matcher import match_docs


class AssignDocsStep:
    """Step 2 — Validate and finalise doc_id assignments for each seed.

    Although doc_ids are pre-assigned in Step 1 via match_docs, this step
    re-validates the assignments, logs warnings for seeds with no matching
    documents (acceptable only for no_action), and ensures no stale doc_ids
    reference non-existent documents.

    Args:
        valid_doc_ids: Set of all doc_ids that exist in the generated corpus.
    """

    def __init__(self, valid_doc_ids: set[str]) -> None:
        self._valid_doc_ids = valid_doc_ids

    def execute(self, seeds: list[QuestionSeed]) -> list[QuestionSeed]:
        """Validate and clean doc_id assignments across all seeds.

        Removes any doc_ids not present in the valid corpus and warns on
        unexpected empty assignments.

        Args:
            seeds: Seeds produced by GenerateSeedsStep.

        Returns:
            Same seed list with doc_ids validated and filtered.
        """
        validated: list[QuestionSeed] = []
        for seed in seeds:
            cleaned_ids = [d for d in seed.doc_ids if d in self._valid_doc_ids]

            if len(cleaned_ids) != len(seed.doc_ids):
                removed = set(seed.doc_ids) - set(cleaned_ids)
                logger.warning(f"{seed.seed_id}: removed invalid doc_ids {removed}")

            if not cleaned_ids and seed.tool_name != "no_action":
                logger.warning(f"{seed.seed_id} ({seed.tool_name}): no valid doc_ids assigned")

            validated.append(seed.model_copy(update={"doc_ids": cleaned_ids}))

        logger.info(f"Doc assignment validated for {len(validated)} seeds")
        return validated
