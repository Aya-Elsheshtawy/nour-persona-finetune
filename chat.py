"""
CPU inference for the Nour persona chatbot — plan §5 (Option A, via Ollama)
+ §5.1's retrieval gate, combined.

Generation is delegated to a local Ollama server (`ollama create` + `ollama
serve`, already running as part of a normal Ollama install) instead of
loading the GGUF directly through llama-cpp-python — simpler on Windows
since Ollama already handles GGUF loading/quantization/serving.

Two layers enforce scope:
  1. A FAISS retrieval gate: scores the incoming message against every
     training question two ways — standalone, and concatenated with the
     previous assistant reply (so short context-dependent follow-ups like
     "why?" aren't scored against the user message alone) — and takes
     whichever score is higher. This matters because concatenation can
     backfire: a complete, self-contained new question ("explain RAG to
     me") can score *lower* once diluted by an unrelated previous answer,
     so the standalone score acts as a floor that a topic change can't be
     dragged below. Below SCOPE_THRESHOLD on both, generation is skipped
     entirely and a fixed deflection is returned — Ollama alone has no
     equivalent of this guardrail.
  2. The fine-tuned model itself, for tone/phrasing of in-scope answers and
     in-scope refusals (privacy boundaries).

Setup (after training, once you have the .gguf from colab/train_nour.ipynb):
    python generate_modelfile.py nour-q4_k_m.gguf
    ollama create nour -f Modelfile

Usage:
    python chat.py --model nour
"""

import argparse

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from data.augmented_data import (
    OUT_OF_SCOPE_AUGMENTED,
    PARAPHRASES_BY_CATEGORY,
    PRIVACY_AUGMENTED,
)
from data.persona_data import CONVERSATIONS, EM_PROMPT

SCOPE_THRESHOLD = 0.55
FALLBACK = "That's outside what I can help with — I can only talk about myself, my work, and my expertise."


def all_training_questions() -> list[str]:
    questions = [q for q, _ in CONVERSATIONS]
    questions += [q for q, _ in OUT_OF_SCOPE_AUGMENTED]
    questions += [q for q, _ in PRIVACY_AUGMENTED]
    for pairs in PARAPHRASES_BY_CATEGORY.values():
        questions += [q for q, _ in pairs]
    return questions


class ScopeGate:
    def __init__(self, threshold: float = SCOPE_THRESHOLD):
        self.threshold = threshold
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        questions = all_training_questions()
        vectors = self.embedder.encode(questions, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.array(vectors, dtype="float32"))

    def _score(self, query: str) -> float:
        vec = self.embedder.encode([query], normalize_embeddings=True)
        score, _ = self.index.search(np.array(vec, dtype="float32"), k=1)
        return float(score[0][0])

    def is_in_scope(self, user_msg: str, last_assistant_msg: str | None = None) -> bool:
        best = self._score(user_msg)
        if last_assistant_msg:
            best = max(best, self._score(f"{last_assistant_msg} {user_msg}"))
        return best >= self.threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nour", help="Ollama model name (from `ollama create <name> -f Modelfile`)")
    parser.add_argument("--threshold", type=float, default=SCOPE_THRESHOLD)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("Loading scope gate (embedding model + FAISS index)...")
    gate = ScopeGate(threshold=args.threshold)

    messages = [{"role": "system", "content": EM_PROMPT}]
    last_assistant_msg = None

    print(f"Nour is ready (model: {args.model}, via Ollama). Type 'exit' to quit.\n")
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_msg})

        if not gate.is_in_scope(user_msg, last_assistant_msg):
            reply = FALLBACK
        else:
            response = ollama.chat(
                model=args.model,
                messages=messages,
                options={"temperature": args.temperature},
            )
            reply = response["message"]["content"]
            last_assistant_msg = reply  # only real answers become context for the next gate check

        print(f"Nour: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
