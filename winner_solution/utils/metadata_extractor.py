"""Regex-based entity metadata extractor for MASA document chunks."""

import re

from winner_solution.schemas.chunk import ChunkMetadata


MODULE_PATTERNS = [
    r'\bcondor\b', r'\bcóndor\b',
    r'\bquetzal\b',
    r'\bjaguar\b',
    r'\bcolibri\b', r'\bcolibrí\b',
    r'\bvicuna\b', r'\bvicuña\b',
    r'\btucan\b', r'\btucán\b',
]

MODULE_CANONICAL: dict[str, str] = {
    'condor': 'condor', 'cóndor': 'condor',
    'quetzal': 'quetzal',
    'jaguar': 'jaguar',
    'colibri': 'colibri', 'colibrí': 'colibri',
    'vicuna': 'vicuna', 'vicuña': 'vicuna',
    'tucan': 'tucan', 'tucán': 'tucan',
}

CREW_NAMES = [
    "Santiago Reyes",
    "Ana Valdivia",
    "Kai Nakamura",
    "Fatima Al-Hassan",
    "Fátima Al-Hassan",
    "Pavel Kozlov",
    "Lucia Mendoza",
    "Lucía Mendoza",
]

CREW_CANONICAL: dict[str, str] = {
    "Santiago Reyes": "Santiago Reyes",
    "Ana Valdivia": "Ana Valdivia",
    "Kai Nakamura": "Kai Nakamura",
    "Fatima Al-Hassan": "Fatima Al-Hassan",
    "Fátima Al-Hassan": "Fatima Al-Hassan",
    "Pavel Kozlov": "Pavel Kozlov",
    "Lucia Mendoza": "Lucia Mendoza",
    "Lucía Mendoza": "Lucia Mendoza",
}

PROTOCOL_PATTERN = re.compile(r'MASA-SEC-\d{3}')

THRESHOLD_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*(?:°C|°F|K|kPa|Pa|bar|ppm|%|kg|g|W|kW|rpm|m/s|km/h|Gy|mSv|lux)'
)

DOCUMENT_TYPE_MAP: dict[tuple[int, int], str] = {
    (1, 6): 'module_spec',
    (7, 12): 'security_protocol',
    (13, 17): 'systems_guide',
    (18, 22): 'operational_procedure',
    (23, 29): 'mission_report',
    (30, 35): 'crew_profile',
    (36, 36): 'systems_overview',
    (37, 37): 'regulations',
    (38, 38): 'quick_reference',
    (39, 39): 'supply_management',
    (40, 40): 'navigation_guide',
    (41, 41): 'research_program',
    (42, 42): 'health_program',
    (50, 61): 'general_information',
}


def _infer_document_type(doc_id: str) -> str:
    """Infer document category from its numeric ID.

    Args:
        doc_id: Document identifier in MASA-DOC-XXX format.

    Returns:
        Document type string, or 'unknown' if not matched.
    """
    match = re.search(r'(\d+)$', doc_id)
    if not match:
        return 'unknown'
    num = int(match.group(1))
    for (lo, hi), doc_type in DOCUMENT_TYPE_MAP.items():
        if lo <= num <= hi:
            return doc_type
    return 'general_information'


class MetadataExtractor:
    """Extracts entity metadata from MASA document chunk text.

    Uses regex patterns to find modules, protocols, crew members, and
    threshold values. All extraction is case-insensitive where appropriate.
    """

    def extract(self, content: str, doc_id: str) -> ChunkMetadata:
        """Extract structured metadata from chunk content.

        Args:
            content: Raw text body of the chunk.
            doc_id: Document identifier used to infer document_type.

        Returns:
            ChunkMetadata with all extracted entities.
        """
        lower = content.lower()

        modules = []
        seen_modules: set[str] = set()
        for pattern in MODULE_PATTERNS:
            for m in re.finditer(pattern, lower):
                raw = m.group(0)
                canonical = MODULE_CANONICAL.get(raw)
                if canonical and canonical not in seen_modules:
                    seen_modules.add(canonical)
                    modules.append(canonical)

        protocols = sorted(set(PROTOCOL_PATTERN.findall(content)))

        crew = []
        seen_crew: set[str] = set()
        for name in CREW_NAMES:
            if name in content or name.lower() in lower:
                canonical = CREW_CANONICAL.get(name, name)
                if canonical not in seen_crew:
                    seen_crew.add(canonical)
                    crew.append(canonical)

        thresholds = list(dict.fromkeys(THRESHOLD_PATTERN.findall(content)))

        return ChunkMetadata(
            modules_mentioned=modules,
            protocols_mentioned=protocols,
            crew_mentioned=crew,
            thresholds=thresholds,
            document_type=_infer_document_type(doc_id),
        )


metadata_extractor = MetadataExtractor()
