"""Fine-tune bge-small-en-v1.5 on consultas_centro_control.json.

Uses MultipleNegativesRankingLoss with hard negatives from the
consultas file. The fine-tuned encoder learns to associate operator
queries with the correct document chunks.

Usage:
    python -m app.winner_solution.scripts.finetune_encoder --device cuda
    python -m app.winner_solution.scripts.finetune_encoder --device cuda --epochs 5 --batch-size 16
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from winner_solution.utils.chunker import chunk_all_documents

PROJECT_ROOT = Path(__file__).parent.parent
CONSULTAS_PATH = PROJECT_ROOT / "data" / "consultas_centro_control.json"
DOCS_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "winner" / "encoder"

ENCODER_MODEL = "BAAI/bge-small-en-v1.5"


def build_training_examples(consultas: list[dict], chunks: list[dict]) -> list[InputExample]:
    """Build training examples from consultas and document chunks.

    For each (query, doc_id) pair, finds the best matching chunk from
    the assigned document. If a hard_negative_doc_id is present, includes
    the best chunk from that doc as a hard negative.

    Args:
        consultas: List of consulta dicts with query, doc_id, hard_negative_doc_id.
        chunks: List of all chunk dicts with doc_id and embedding_text.

    Returns:
        List of InputExample objects for sentence-transformers training.
    """
    chunks_by_doc = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)

    examples = []

    for consulta in consultas:
        query = consulta["query"]
        pos_doc_id = consulta["doc_id"]
        neg_doc_id = consulta.get("hard_negative_doc_id")

        pos_chunks = chunks_by_doc.get(pos_doc_id, [])
        if not pos_chunks:
            continue

        pos_text = pos_chunks[0]["embedding_text"]
        best_overlap = 0
        query_words = set(query.lower().split())
        for chunk in pos_chunks:
            overlap = len(query_words & set(chunk["content"].lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                pos_text = chunk["embedding_text"]

        if neg_doc_id:
            neg_chunks = chunks_by_doc.get(neg_doc_id, [])
            if neg_chunks:
                neg_text = neg_chunks[0]["embedding_text"]
                examples.append(InputExample(texts=[query, pos_text, neg_text]))
                continue

        examples.append(InputExample(texts=[query, pos_text]))

    return examples


def build_evaluator(consultas: list[dict], chunks: list[dict]) -> InformationRetrievalEvaluator:
    """Build an IR evaluator for validation during training.

    Uses a subset of consultas as queries and all chunk embedding_texts
    as the corpus. Measures recall@5 and MRR.

    Args:
        consultas: List of consulta dicts.
        chunks: List of all chunk dicts.

    Returns:
        Configured InformationRetrievalEvaluator.
    """
    queries = {}
    relevant_docs = {}
    corpus = {}

    for i, chunk in enumerate(chunks):
        corpus[str(i)] = chunk["embedding_text"]

    chunk_to_idx = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk["doc_id"]
        if doc_id not in chunk_to_idx:
            chunk_to_idx[doc_id] = []
        chunk_to_idx[doc_id].append(str(i))

    for i, consulta in enumerate(consultas[:100]):
        qid = f"q{i}"
        queries[qid] = consulta["query"]
        doc_id = consulta["doc_id"]
        relevant_docs[qid] = set(chunk_to_idx.get(doc_id, []))

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="consultas_eval",
        mrr_at_k=[5],
        recall_at_k=[1, 3, 5],
        show_progress_bar=False,
    )


def main() -> None:
    """Fine-tune bge-small-en-v1.5 on consultas with hard negatives."""
    args = sys.argv[1:]
    device = "cuda"
    epochs = 3
    batch_size = 16
    lr = 2e-5

    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--epochs" in args:
        epochs = int(args[args.index("--epochs") + 1])
    if "--batch-size" in args:
        batch_size = int(args[args.index("--batch-size") + 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = SentenceTransformer(ENCODER_MODEL, device=device)

    print("Chunking documents...")
    print(f"Total chunks: {len(chunks)}")

    print("Loading consultas...")
    consultas = json.loads(CONSULTAS_PATH.read_text(encoding="utf-8"))
    print(f"Total consultas: {len(consultas)}")

    print("Building training examples...")
    examples = build_training_examples(consultas, chunks)
    has_negatives = sum(1 for e in examples if len(e.texts) == 3)
    print(f"Training examples: {len(examples)} ({has_negatives} with hard negatives)")

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Building evaluator...")
    evaluator = build_evaluator(consultas, chunks)

    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        evaluation_steps=len(train_dataloader),
        output_path=str(OUTPUT_DIR / "finetuned_encoder"),
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    print(f"Fine-tuned encoder saved to {OUTPUT_DIR / 'finetuned_encoder'}")


if __name__ == "__main__":
    main()
