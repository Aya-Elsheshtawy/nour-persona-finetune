"""
Convert the Nour persona data into ChatML-style instruction-tuning JSONL,
per §1.1 of nour_persona_finetune_plan.md.

Combines:
  - data/persona_data.py       (154 original single-turn Q&A pairs)
  - data/augmented_data.py     (paraphrases, oversampled boundary examples,
                                 and multi-turn conversations, per §1.2)

Splits 85% train / 15% val, stratified by category so validation isn't
skewed toward any one topic, and writes train.jsonl / val.jsonl.
"""

import json
import random
from pathlib import Path

from data.persona_data import EM_PROMPT, CONVERSATIONS_BY_CATEGORY
from data.augmented_data import (
    OUT_OF_SCOPE_AUGMENTED,
    PRIVACY_AUGMENTED,
    PARAPHRASES_BY_CATEGORY,
    MULTI_TURN_CONVERSATIONS,
)

SEED = 42
VAL_FRACTION = 0.15
OUTPUT_DIR = Path(__file__).parent / "data"


def build_examples_by_category() -> dict[str, list[list[dict]]]:
    """Return {category: [messages, ...]} where each `messages` is the full
    system+user+assistant turn sequence for one training example."""
    by_category: dict[str, list[list[dict]]] = {}

    def add_pair(category: str, question: str, answer: str) -> None:
        by_category.setdefault(category, []).append([
            {"role": "system", "content": EM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

    for category, pairs in CONVERSATIONS_BY_CATEGORY.items():
        for question, answer in pairs:
            add_pair(category, question, answer)

    for category, pairs in PARAPHRASES_BY_CATEGORY.items():
        for question, answer in pairs:
            add_pair(category, question, answer)

    for question, answer in OUT_OF_SCOPE_AUGMENTED:
        add_pair("out_of_scope", question, answer)

    for question, answer in PRIVACY_AUGMENTED:
        add_pair("privacy_boundaries", question, answer)

    multi_turn_examples = []
    for turns in MULTI_TURN_CONVERSATIONS:
        multi_turn_examples.append([{"role": "system", "content": EM_PROMPT}, *turns])
    by_category["multi_turn"] = multi_turn_examples

    return by_category


def stratified_split(by_category: dict[str, list[list[dict]]], val_fraction: float, seed: int):
    rng = random.Random(seed)
    train, val = [], []

    for category, examples in by_category.items():
        shuffled = examples[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) >= 4 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, examples: list[list[dict]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for messages in examples:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


def main() -> None:
    by_category = build_examples_by_category()
    total = sum(len(v) for v in by_category.values())

    train, val = stratified_split(by_category, VAL_FRACTION, SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "train.jsonl", train)
    write_jsonl(OUTPUT_DIR / "val.jsonl", val)

    print(f"Categories: {len(by_category)}")
    for category, examples in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"  {category:22s} {len(examples)}")
    print(f"Total examples: {total}")
    print(f"Train: {len(train)}  Val: {len(val)}")


if __name__ == "__main__":
    main()
