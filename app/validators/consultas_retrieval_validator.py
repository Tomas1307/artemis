"""Validator that checks consultas_centro_control.json query-doc pairs via embedding retrieval."""

import json
from pathlib import Path

import numpy as np

from app.validators.base_validator import BaseValidator
from app.validators.validation_report import ValidationFinding, ValidationReport

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONSULTAS_PATH = PROJECT_ROOT / "proyecto_artemis" / "datos_entrenamiento" / "consultas_centro_control.json"
DOCS_INDEX_PATH = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento" / "documentos_masa.json"

ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


class ConsultasRetrievalValidator(BaseValidator):
    """Verify each (query, doc_id) pair retrieves its assigned doc in top-K.

    Loads bge-small-en-v1.5, embeds all 54 documents and each of the 800 queries,
    then checks if the assigned doc_id ranks in the top-K by cosine similarity.
    Flags rank > 3 as error, rank 2-3 as warning.
    """

    @property
    def name(self) -> str:
        """Return the validator name."""
        return "ConsultasRetrievalValidator"

    def validate(self) -> ValidationReport:
        """Run embedding-based retrieval validation on consultas_centro_control."""
        from sentence_transformers import SentenceTransformer

        consultas = json.loads(CONSULTAS_PATH.read_text(encoding="utf-8"))
        docs_index = json.loads(DOCS_INDEX_PATH.read_text(encoding="utf-8"))
        doc_ids = sorted(docs_index.keys())
        doc_texts = self._load_doc_texts(docs_index, doc_ids)

        model = SentenceTransformer(ENCODER_MODEL)

        doc_embeddings = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)
        query_texts = [c["query"] for c in consultas]
        query_embeddings = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)

        similarities = np.dot(query_embeddings, doc_embeddings.T)

        findings: list[ValidationFinding] = []
        passed = 0
        rank_distribution = {1: 0}

        for idx, consulta in enumerate(consultas):
            assigned_doc = consulta["doc_id"]
            if assigned_doc not in doc_ids:
                findings.append(ValidationFinding(
                    severity="error",
                    rule="doc_id_not_in_corpus",
                    message=f"Assigned doc_id '{assigned_doc}' not found in document corpus.",
                    context={"query": consulta["query"][:100]},
                ))
                continue

            doc_idx = doc_ids.index(assigned_doc)
            scores = similarities[idx]
            ranked_indices = np.argsort(-scores)
            rank = int(np.where(ranked_indices == doc_idx)[0][0]) + 1

            if rank == 1:
                passed += 1
                rank_distribution[1] = rank_distribution.get(1, 0) + 1
            elif rank <= 3:
                findings.append(ValidationFinding(
                    severity="warning",
                    rule="doc_not_rank1",
                    message=f"Assigned doc '{assigned_doc}' ranked #{rank} (not #1).",
                    context={
                        "query": consulta["query"][:150],
                        "assigned_doc": assigned_doc,
                        "rank": rank,
                        "top3_docs": [doc_ids[i] for i in ranked_indices[:3]],
                        "top3_scores": [round(float(scores[i]), 4) for i in ranked_indices[:3]],
                    },
                ))
            else:
                top5_docs = [doc_ids[i] for i in ranked_indices[:TOP_K]]
                top5_scores = [round(float(scores[i]), 4) for i in ranked_indices[:TOP_K]]
                findings.append(ValidationFinding(
                    severity="error",
                    rule="doc_not_top3",
                    message=f"Assigned doc '{assigned_doc}' ranked #{rank} (not in top-3). Top-3: {top5_docs[:3]}",
                    context={
                        "query": consulta["query"][:150],
                        "assigned_doc": assigned_doc,
                        "rank": rank,
                        "top5_docs": top5_docs,
                        "top5_scores": top5_scores,
                    },
                ))

        return ValidationReport(
            validator_name=self.name,
            total_checked=len(consultas),
            passed=passed,
            findings=findings,
        )

    def _load_doc_texts(self, docs_index: dict, doc_ids: list[str]) -> list[str]:
        """Load the text content of each document in order of doc_ids."""
        texts = []
        for doc_id in doc_ids:
            doc_info = docs_index[doc_id]
            doc_path = PROJECT_ROOT / "proyecto_artemis" / "base_conocimiento" / doc_id / "doc.md"
            if doc_path.exists():
                content = doc_path.read_text(encoding="utf-8")
                texts.append(content[:8000])
            else:
                texts.append(f"Document {doc_id}: {doc_info.get('title', 'Unknown')}")
        return texts
