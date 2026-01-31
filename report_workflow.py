# how to run: 
# python report_workflow.py path/to/report.pdf --out-dir path/to/outdir
# by default, outputs will be written to ./out/

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import fitz  # PyMuPDF
from transformers import pipeline


# ----------------------------
# Config
# ----------------------------

@dataclass
class WorkflowConfig:
    pdf_path: Path
    out_dir: Path
    translate_model: str = "Helsinki-NLP/opus-mt-fi-en"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    device: int = -1
    translate_batch_size: int = 32
    emotion_batch_size: int = 32
    max_chunk_chars: int = 400


# ----------------------------
# Stage 1: Extract + sentence split
# ----------------------------

_ABBREV = [
    "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.",
    "e.g.", "i.e.", "etc.", "vs.", "No.", "Fig.", "pp.",
    "Inc.", "Ltd.", "Co.", "Corp.", "St.", "EU", "EUR", "Tel.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.",
    "Sept.", "Oct.", "Nov.", "Dec."
]


def extract_pages_pymupdf(pdf_path: Path) -> List[Dict[str, object]]:
    pages: List[Dict[str, object]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append({"page": i, "text": text})
    return pages


def normalize_text(s: str) -> str:
    # Fix hyphenation at line breaks: "coro-\nnavirus" -> "coronavirus"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # Convert newlines/tabs to spaces
    s = re.sub(r"[\t\r\n]+", " ", s)
    # Collapse repeated spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def split_sentences(text: str) -> List[str]:
    if not text:
        return []

    placeholder = "<DOT>"
    protected = text
    for ab in _ABBREV:
        protected = protected.replace(ab, ab.replace(".", placeholder))

    parts = re.split(r"(?<=[.!?])\s+", protected)

    sents: List[str] = []
    for p in parts:
        p = p.replace(placeholder, ".").strip()
        if len(p) >= 2:
            sents.append(p)
    return sents


def make_sentence_records(pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for p in pages:
        page_no = int(p["page"])
        text_norm = normalize_text(str(p["text"]))
        sents = split_sentences(text_norm)
        for idx, sent in enumerate(sents, start=1):
            records.append({
                "page": page_no,
                "sent_id_on_page": idx,
                "sentence_fi": sent
            })
    return records


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sentences_txt(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"[p.{int(r['page']):03d}] {r['sentence_fi']}\n")


# ----------------------------
# Stage 2: Translation (FI -> EN)
# ----------------------------

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def chunk_text(s: str, max_chars: int) -> List[str]:
    """
    Chunk by characters; tries to split on punctuation, else hard splits.
    """
    s = s.strip()
    if len(s) <= max_chars:
        return [s]

    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        boundary = max(
            s.rfind(".", start, end),
            s.rfind("!", start, end),
            s.rfind("?", start, end),
        )
        if boundary <= start + 50:
            boundary = end
        else:
            boundary = boundary + 1
        chunks.append(s[start:boundary].strip())
        start = boundary
    return [c for c in chunks if c]


def build_translator(model_name: str, device: int):
    return pipeline("translation", model=model_name, device=device)


def translate_sentences(
    translator,
    sentences: List[str],
    batch_size: int,
    max_chunk_chars: int,
) -> List[str]:
    out: List[str] = []
    n = len(sentences)
    n_batches = ceil(n / batch_size)

    for b in range(n_batches):
        batch = sentences[b*batch_size:(b+1)*batch_size]
        batch = [normalize_spaces(x) for x in batch]

        # Prepare chunking
        prepared: List[object] = []
        for s in batch:
            chunks = chunk_text(s, max_chars=max_chunk_chars)
            prepared.append(chunks if len(chunks) > 1 else chunks[0])

        # Flatten for pipeline
        flat_inputs: List[str] = []
        flat_map: List[int] = []
        for i, item in enumerate(prepared):
            if isinstance(item, list):
                for c in item:
                    flat_inputs.append(c)
                    flat_map.append(i)
            else:
                flat_inputs.append(item)
                flat_map.append(i)

        preds = translator(flat_inputs)

        parts: List[List[str]] = [[] for _ in range(len(batch))]
        for pred, idx in zip(preds, flat_map):
            parts[idx].append(pred["translation_text"])

        for i in range(len(batch)):
            out.append(" ".join(parts[i]).strip())

        print(f"Translated batch {b+1}/{n_batches} ({len(out)}/{n})")

    return out


# ----------------------------
# Stage 3: Emotion analysis
# ----------------------------

def build_emotion_pipe(model_name: str, device: int):
    return pipeline(
        task="text-classification",
        model=model_name,
        device=device,
        top_k=None,
        truncation=True,
    )


def analyze_emotions(
    emo_pipe,
    texts: List[str],
    batch_size: int,
) -> Tuple[List[str], List[float], List[str]]:
    top_labels: List[str] = []
    top_scores: List[float] = []
    scores_json: List[str] = []

    n = len(texts)
    n_batches = ceil(n / batch_size)

    for b in range(n_batches):
        batch = texts[b*batch_size:(b+1)*batch_size]
        batch = [("" if pd.isna(t) else str(t)) for t in batch]

        preds = emo_pipe(batch)

        for label_scores in preds:
            d = {x["label"]: float(x["score"]) for x in label_scores} if label_scores else {}
            best_label, best_score = max(d.items(), key=lambda kv: kv[1]) if d else ("", 0.0)

            top_labels.append(best_label)
            top_scores.append(best_score)
            scores_json.append(json.dumps(d, ensure_ascii=False))

        print(f"Emotion batch {b+1}/{n_batches} ({len(top_labels)}/{n})")

    return top_labels, top_scores, scores_json


def expand_emotions_wide(df: pd.DataFrame, json_col: str = "emotion_scores_json") -> pd.DataFrame:
    def json_to_dict(x):
        if pd.isna(x):
            return {}
        if isinstance(x, dict):
            return x
        return json.loads(x)

    scores_dicts = df[json_col].apply(json_to_dict)
    emo_df = pd.json_normalize(scores_dicts).add_prefix("emo_")
    out = pd.concat([df.drop(columns=[json_col]), emo_df], axis=1).fillna(0.0)
    return out


# ----------------------------
# Orchestrator
# ----------------------------

def run_workflow(cfg: WorkflowConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {cfg.pdf_path}")

    # --- Stage 1 outputs
    stem = cfg.pdf_path.stem
    sentences_jsonl = cfg.out_dir / f"{stem}.sentences.jsonl"
    sentences_txt = cfg.out_dir / f"{stem}.sentences.txt"
    sentences_csv = cfg.out_dir / f"{stem}.sentences_fi_en.csv"
    emotions_csv = cfg.out_dir / f"{stem}.sentences_fi_en_emotions.csv"
    emotions_wide_csv = cfg.out_dir / f"{stem}.sentences_fi_en_emotions_wide.csv"

    # 1) Extract
    pages = extract_pages_pymupdf(cfg.pdf_path)
    records = make_sentence_records(pages)
    write_jsonl(sentences_jsonl, records)
    write_sentences_txt(sentences_txt, records)
    print(f"Wrote: {sentences_jsonl}")
    print(f"Wrote: {sentences_txt}")

    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    df["line_id"] = df.index + 1
    df["sentence_fi"] = df["sentence_fi"].astype(str).map(normalize_spaces)

    # 2) Translate
    translator = build_translator(cfg.translate_model, cfg.device)
    df["sentence_en"] = translate_sentences(
        translator,
        df["sentence_fi"].tolist(),
        batch_size=cfg.translate_batch_size,
        max_chunk_chars=cfg.max_chunk_chars,
    )

    df_out = df[["line_id", "page", "sent_id_on_page", "sentence_fi", "sentence_en"]]
    df_out.to_csv(sentences_csv, index=False, encoding="utf-8")
    print(f"Wrote: {sentences_csv}")

    # 3) Emotions
    emo_pipe = build_emotion_pipe(cfg.emotion_model, cfg.device)
    top_emotion, top_score, scores_json = analyze_emotions(
        emo_pipe,
        df_out["sentence_en"].tolist(),
        batch_size=cfg.emotion_batch_size,
    )
    df_out["top_emotion"] = top_emotion
    df_out["top_emotion_score"] = top_score
    df_out["emotion_scores_json"] = scores_json

    df_out.to_csv(emotions_csv, index=False, encoding="utf-8")
    print(f"Wrote: {emotions_csv}")

    df_wide = expand_emotions_wide(df_out, json_col="emotion_scores_json")
    df_wide.to_csv(emotions_wide_csv, index=False, encoding="utf-8")
    print(f"Wrote: {emotions_wide_csv}")

    print("Done.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> WorkflowConfig:
    p = argparse.ArgumentParser(
        description="Extract PDF sentences, translate fi->en, run emotion analysis, and export CSVs."
    )
    p.add_argument("pdf", type=Path, help="Path to the input PDF.")
    p.add_argument("--out-dir", type=Path, default=Path("out"), help="Output directory.")
    p.add_argument("--translate-model", type=str, default="Helsinki-NLP/opus-mt-fi-en")
    p.add_argument("--emotion-model", type=str, default="SamLowe/roberta-base-go_emotions")
    p.add_argument("--translate-batch-size", type=int, default=32)
    p.add_argument("--emotion-batch-size", type=int, default=32)
    p.add_argument("--max-chunk-chars", type=int, default=400)

    a = p.parse_args()

    device = 0 if torch.cuda.is_available() else -1

    return WorkflowConfig(
        pdf_path=a.pdf,
        out_dir=a.out_dir,
        translate_model=a.translate_model,
        emotion_model=a.emotion_model,
        device=device,
        translate_batch_size=a.translate_batch_size,
        emotion_batch_size=a.emotion_batch_size,
        max_chunk_chars=a.max_chunk_chars,
    )


def main() -> None:
    cfg = parse_args()
    run_workflow(cfg)


if __name__ == "__main__":
    main()
