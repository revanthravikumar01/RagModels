"""
Evaluate the chatbot on the questions in tests/eval_questions.json.

Usage (PowerShell, with the virtual environment active):
    python evaluate.py           # retrieval only (fast, about 1 minute)
    python evaluate.py --full    # also generate answers with the LLM (slower)

Metrics:
  Hit rate      - answerable questions where a correct page is in the evidence
  MRR           - Mean Reciprocal Rank: 1.0 if the correct page is ranked first,
                  0.5 if second, and so on (higher is better)
  Refusals      - off-topic questions that were correctly refused
  Repetition    - (--full only) answers with a line repeated, should be 0
"""

import json
import sys
import time
from pathlib import Path

import backend

QUESTIONS_FILE = Path(__file__).resolve().parent / "tests" / "eval_questions.json"


def is_expected(kb: backend.KnowledgeBase, chunk: dict, expect: dict) -> bool:
    name = kb.documents[chunk["doc_id"]]["display_name"]
    return chunk.get("page") in expect.get(name, [])


def has_repeated_line(text: str) -> bool:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    return len(lines) != len(set(lines))


def main() -> None:
    full = "--full" in sys.argv
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]

    print("Loading models and index...")
    kb = backend.KnowledgeBase()
    threshold = backend.DEFAULT_MIN_RELEVANCE
    print(f"Embedding model: {backend.EMBEDDING_MODEL_NAME} | re-ranker: {backend.RERANKER_MODEL_NAME} "
          f"| min relevance: {threshold}\n")

    answerable = refusable = hits = refused_ok = repeated = 0
    reciprocal_ranks = []
    start = time.time()

    for q in questions:
        if full:
            result = kb.answer(q["question"], previous_question=q.get("previous"))
            refused = result.status == "insufficient"
            if has_repeated_line(result.answer):
                repeated += 1
        relevant, query_info = kb.search(q["question"], previous_question=q.get("previous"))
        best = query_info.get("best_relevance", 0.0)
        if best < threshold:
            relevant = []
        if not full:
            refused = not relevant

        if q.get("refuse"):
            refusable += 1
            ok = refused
            refused_ok += ok
            detail = "refused" if refused else f"answered (best relevance {best:.3f})"
        else:
            answerable += 1
            rank = next((i + 1 for i, h in enumerate(relevant)
                         if is_expected(kb, h["chunk"], q["expect"])), None)
            ok = rank is not None and not (full and refused)
            hits += ok
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            detail = (f"correct page at rank {rank}" if rank else "correct page NOT found") + f" (best relevance {best:.3f})"
            if full and refused:
                detail += " but the LLM refused"

        print(f"{'PASS' if ok else 'FAIL'}  {q['id']:<8} {q['question'][:55]:<55} {detail}")
        if full and not q.get("refuse"):
            first_line = result.answer.strip().splitlines()[0] if result.answer.strip() else ""
            print(f"      model={result.model} | {first_line[:110]}")

    mrr = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    print("\n" + "=" * 70)
    print(f"Hit rate:   {hits}/{answerable} = {hits / max(answerable, 1):.0%}")
    print(f"MRR:        {mrr:.2f}")
    print(f"Refusals:   {refused_ok}/{refusable} = {refused_ok / max(refusable, 1):.0%}")
    if full:
        print(f"Repetition: {repeated} answer(s) with a repeated line")
    print(f"Time:       {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
