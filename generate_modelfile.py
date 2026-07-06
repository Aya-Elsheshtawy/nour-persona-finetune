"""
Generate an Ollama Modelfile from the same EM_PROMPT used at training time
(data/persona_data.py), so the inference-time system prompt never drifts
from the training-time one — the plan (§1.4) calls this out as an easy,
easy-to-miss cause of persona/scope drift.

Note: running the resulting model straight through `ollama run` skips the
FAISS scope gate (§5.1) — that only runs inside chat.py. Use Ollama for
quick manual testing, chat.py for the guardrail-enforced experience.

Usage:
    python generate_modelfile.py nour-q4_k_m.gguf
"""

import sys
from pathlib import Path

from data.persona_data import EM_PROMPT


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python generate_modelfile.py <path-to-gguf>")
        sys.exit(1)

    gguf_path = sys.argv[1]
    modelfile = (
        f'FROM {gguf_path}\n'
        f'SYSTEM """{EM_PROMPT}"""\n'
        f'PARAMETER temperature 0.7\n'
        f'PARAMETER stop "<|im_end|>"\n'
    )
    Path("Modelfile").write_text(modelfile, encoding="utf-8")
    print("Wrote Modelfile. Now run:\n  ollama create nour -f Modelfile\n  ollama run nour")


if __name__ == "__main__":
    main()
