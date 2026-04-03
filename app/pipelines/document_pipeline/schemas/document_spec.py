from pydantic import BaseModel


class DocumentSpec(BaseModel):
    """Specification for a single document to generate.

    Loaded from document_registry.yaml. Each spec defines what the document
    covers, what skeleton data it must contain, and the target length.

    Attributes:
        doc_id: Unique document identifier (e.g., MASA-DOC-001).
        title: Human-readable document title.
        type: Document category (module_manual, protocol_group, system_guide,
            operational_procedure, mission_record, crew_profile, cross_cutting, noise).
        target_words: Target word count for the generated document.
        skeleton_refs: List of dotted paths into the skeleton YAML that this
            document must reference. Empty for noise documents.
        sections: Ordered list of section headings the document must include.
    """

    doc_id: str
    title: str
    type: str
    target_words: int
    skeleton_refs: list[str]
    sections: list[str]
