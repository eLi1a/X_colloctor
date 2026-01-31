#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    input_pdf: Path
    out_dir: Path
    source_lang: str = "auto"
    target_lang: str = "en"
    do_translate: bool = True
    do_emotion: bool = True


# ----------------------------
# PDF extraction
# ----------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF.
    Replace internals with whatever you used in the notebook
    (pdfplumber, pymupdf/fitz, pypdf, etc.).
    """
    # Example using PyMuPDF (fitz):
    # import fitz
    # doc = fitz.open(pdf_path)
    # text = "\n".join(page.get_text("text") for page in doc)
    # return text

    raise NotImplementedError("Wire this to your notebook's PDF extraction code.")


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Simple chunker to avoid overloading translators / emotion models."""
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for line in text.splitlines():
        # keep line breaks gently
        add = (line + "\n")
        if size + len(add) > max_chars and buf:
            chunks.append("".join(buf).strip())
            buf, size = [], 0
        buf.append(add)
        size += len(add)
    if buf:
        chunks.append("".join(buf).strip())
    return [c for c in chunks if c]


# ----------------------------
# Translation
# ----------------------------

def translate_chunks(chunks: List[str], source_lang: str, target_lang: str) -> List[str]:
    """
    Translate text chunks.
    Replace internals with your notebook translator
    (DeepL, googletrans, transformers, OpenAI, etc.).
    """
    # Example placeholder:
    # from deep_translator import GoogleTranslator
    # tr = GoogleTranslator(source=source_lang, target=target_lang)
    # return [tr.translate(c) for c in chunks]

    raise NotImplementedError("Wire this to your notebook's translation code.")


# ----------------------------
# Emotion analysis
# ----------------------------

def emotion_scores(chunks: List[str]) -> List[Dict[str, float]]:
    """
    Run emotion analysis per chunk.
    Replace internals with your notebook model (e.g., transformers pipeline).
    """
    # Example (HuggingFace):
    # from transformers import pipeline
    # clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    # out = clf(chunks)
    # out is list[list[dict(label, score)]]
    # return [{d["label"]: float(d["score"]) for d in one} for one in out]

    raise NotImplementedError("Wire this to your notebook's emotion analysis code.")


def aggregate_emotions(per_chunk: List[Dict[str, float]]) -> Dict[str, float]:
    """Average scores across chunks."""
    if not per_chunk:
        return {}
    keys = set().union(*per_chunk)
    agg: Dict[str, float] = {}
    for k in keys:
        agg[k] = sum(d.get(k, 0.0) for d in per_chunk) / len(per_chunk)
    return dict(sorted(agg.items(), key=lambda x: x[1], reverse=True))


# ----------------------------
# I/O helpers
# ----------------------------

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# Pipeline
# ----------------------------

def run(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = extract_text_from_pdf(cfg.input_pdf)
    write_text(cfg.out_dir / "extracted.txt", raw_text)

    chunks = chunk_text(raw_text)

    final_chunks = chunks
    if cfg.do_translate:
        translated = translate_chunks(chunks, cfg.source_lang, cfg.target_lang)
        write_text(cfg.out_dir / "translated.txt", "\n\n".join(translated))
        final_chunks = translated

    if cfg.do_emotion:
        per_chunk_scores = emotion_scores(final_chunks)
        write_json(cfg.out_dir / "emotion_per_chunk.json", per_chunk_scores)
        write_json(cfg.out_dir / "emotion_aggregate.json", aggregate_emotions(per_chunk_scores))


# ----------------------------
# CLI / main
# ----------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Extract PDF text, translate it, and run emotion analysis.")
    p.add_argument("input_pdf", type=Path, help="Path to input PDF.")
    p.add_argument("--out-dir", type=Path, default=Path("out"), help="Output directory.")
    p.add_argument("--source-lang", type=str, default="auto", help="Source language (e.g. auto, fi, zh).")
    p.add_argument("--target-lang", type=str, default="en", help="Target language (e.g. en).")
    p.add_argument("--no-translate", action="store_true", help="Disable translation step.")
    p.add_argument("--no-emotion", action="store_true", help="Disable emotion analysis step.")
    a = p.parse_args()

    return Config(
        input_pdf=a.input_pdf,
        out_dir=a.out_dir,
        source_lang=a.source_lang,
        target_lang=a.target_lang,
        do_translate=not a.no_translate,
        do_emotion=not a.no_emotion,
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
