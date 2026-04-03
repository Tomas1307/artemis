from loguru import logger

from app.pipelines.document_pipeline.schemas.document_spec import DocumentSpec
from app.skeleton.schemas.skeleton_schema import SkeletonSchema
from app.utils.skeleton_extractor import extract_required_facts, format_skeleton_context


class ExtractContextStep:
    """Extracts skeleton data and required facts for a document spec.

    Resolves the document's skeleton_refs against the loaded skeleton,
    producing both a formatted context string for the generation prompt
    and a list of verifiable facts for the judge validation.

    Args:
        skeleton: Validated SkeletonSchema instance.
    """

    def __init__(self, skeleton: SkeletonSchema) -> None:
        self._skeleton = skeleton

    def execute(self, spec: DocumentSpec) -> tuple[str, list[str]]:
        """Extract context and facts for a single document spec.

        Args:
            spec: Document specification with skeleton_refs to resolve.

        Returns:
            Tuple of (skeleton_context_string, required_facts_list).
            For noise documents, returns empty context and empty facts.
        """
        if spec.type == "noise" or not spec.skeleton_refs:
            logger.debug(f"{spec.doc_id}: noise document, no skeleton context")
            return "", []

        skeleton_context = format_skeleton_context(self._skeleton, spec.skeleton_refs)
        required_facts = extract_required_facts(self._skeleton, spec.skeleton_refs)

        logger.info(
            f"{spec.doc_id}: extracted {len(required_facts)} facts "
            f"from {len(spec.skeleton_refs)} skeleton refs"
        )
        return skeleton_context, required_facts
