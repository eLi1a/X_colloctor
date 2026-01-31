
"""
X (Twitter) batch data workflow

What this script does:
1) Combine multiple X batch CSVs into one dataset (schema-checked, de-duplicated)
2) Translate post text (FI -> EN) using Helsinki-NLP translation models (GPU if available)
3) Run emotion analysis (GoEmotions) on translated text
4) Produce simple plots + word clouds (optional)

You can also run this file directly:
    python x_workflow.py --help
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import pipeline


# ----------------------------
# Config
# ----------------------------

DEFAULT_COLUMNS = [
    "x_id",
    "tweet_id",
    "created_at",
    "lang",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "text",
]


@dataclass
class Config:
    input_csvs: List[Path]
    out_dir: Path
    combined_name: str = "x_combined.csv"
    translate_model: str = "Helsinki-NLP/opus-mt-fi-en"
    # If you really want the big model, set via CLI; it is heavier:
    translate_model_big: str = "Helsinki-NLP/opus-mt-tc-big-fi-en"
    use_big_model: bool = False
    translation_batch_size: int = 32
    max_length: int = 512
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    emotion_batch_size: int = 32
    make_plots: bool = False
    make_wordclouds: bool = False


# ----------------------------
# Utilities
# ----------------------------

def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def device_id() -> int:
    """Return HF pipeline device id: 0 for first GPU, else -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def remove_urls(text: str) -> str:
    return re.sub(r"http[s]?://\S+", "", text).strip()


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


# ----------------------------
# 1) Combine datasets
# ----------------------------

def combine_datasets(
    dataset_paths: Iterable[Path],
    output_path: Path,
    columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple CSV datasets row-wise.
    - If `columns` is provided, enforces presence and order.
    - If `columns` is None, uses columns from the first dataset.
    - Drops duplicates by default.

    Returns combined DataFrame and writes it to output_path.
    """
    dataset_paths = [Path(p) for p in dataset_paths]
    if not dataset_paths:
        raise ValueError("No dataset paths provided.")

    frames: List[pd.DataFrame] = []

    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")

        df = pd.read_csv(path)

        if columns is None:
            columns = list(df.columns)

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df[columns]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if drop_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates()
        after = len(combined)
        print(f"De-duplicated: {before} → {after} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Combined {len(dataset_paths)} files → {output_path} ({len(combined)} rows)")

    return combined


# ----------------------------
# 2) Translation
# ----------------------------

def build_translator(model_name: str, device: int):
    # torch_dtype=float16 helps on many NVIDIA GPUs; safe to omit on CPU.
    kwargs = {"device": device}
    if device >= 0:
        kwargs["torch_dtype"] = torch.float16
    return pipeline("translation", model=model_name, **kwargs)


def translate_texts(
    translator,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
) -> List[str]:
    """
    Translate a list of texts using HF pipeline batching.
    Returns list of translated strings (aligned with input order).
    """
    out: List[str] = []
    n = len(texts)
    n_batches = ceil(n / batch_size)

    for b in range(n_batches):
        batch = texts[b * batch_size:(b + 1) * batch_size]
        preds = translator(batch, max_length=max_length)
        out.extend([p["translation_text"] for p in preds])
        print(f"Translated batch {b+1}/{n_batches} ({len(out)}/{n})")

    return out


# ----------------------------
# 3) Emotion analysis
# ----------------------------

def build_emotion_pipe(model_name: str, device: int):
    return pipeline(
        "text-classification",
        model=model_name,
        device=device,
        top_k=None,          # return all labels
        truncation=True,
    )


def analyze_emotions(
    emo_pipe,
    texts: List[str],
    batch_size: int = 32,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Returns:
      top_emotion: list[str]
      top_emotion_score: list[float]
      emotion_scores_json: list[str] (json-serialized dict)
    """
    top_labels: List[str] = []
    top_scores: List[float] = []
    scores_json: List[str] = []

    n = len(texts)
    n_batches = ceil(n / batch_size)

    for b in range(n_batches):
        batch = texts[b * batch_size:(b + 1) * batch_size]
        batch = [safe_text(t) for t in batch]

        preds = emo_pipe(batch)  # list of list-of-dicts
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
    return pd.concat([df.drop(columns=[json_col]), emo_df], axis=1).fillna(0.0)


# ----------------------------
# 4) Optional analysis (plots/wordclouds)
# ----------------------------

def plot_top_emotions(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    import matplotlib.pyplot as plt

    top_counts = df["top_emotion"].value_counts().sort_values()

    plt.figure(figsize=(10, 6))
    top_counts.plot(kind="barh", color="steelblue")
    plt.title("Most Likely Emotion Counts")
    plt.xlabel("Number of posts")
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


def make_wordclouds(df: pd.DataFrame, out_dir: Path, top_k: int = 6) -> None:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    ensure_out_dir(out_dir)

    custom_stopwords = STOPWORDS.union({
        "https", "http", "www", "com", "co", "amp", "t.co", "bit.ly",
        "rt", "via", "ampamp", "ampampamp"
    })

    top_emotions = df["top_emotion"].value_counts().head(top_k).index.tolist()

    for emo in top_emotions:
        texts = df.loc[df["top_emotion"] == emo, "text_clean"].dropna()
        if texts.empty:
            continue

        corpus = " ".join(texts)
        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            max_words=200,
            collocations=False,
            stopwords=custom_stopwords,
        ).generate(corpus)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud – {emo}")
        plt.tight_layout()

        out_path = out_dir / f"wordcloud_{emo}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved wordcloud: {out_path}")


# ----------------------------
# Orchestrator
# ----------------------------

def run(cfg: Config) -> None:
    ensure_out_dir(cfg.out_dir)

    combined_path = cfg.out_dir / cfg.combined_name
    df = combine_datasets(cfg.input_csvs, combined_path, columns=DEFAULT_COLUMNS)

    # Clean/normalize
    df["text"] = df["text"].apply(safe_text).map(normalize_spaces)

    # Translation
    dev = device_id()
    model_to_use = cfg.translate_model_big if cfg.use_big_model else cfg.translate_model
    print(f"Using device: {'cuda:0' if dev >= 0 else 'cpu'}")
    print(f"Translation model: {model_to_use}")

    translator = build_translator(model_to_use, dev)
    df["text_en"] = translate_texts(
        translator,
        df["text"].tolist(),
        batch_size=cfg.translation_batch_size,
        max_length=cfg.max_length,
    )

    translated_path = cfg.out_dir / (combined_path.stem + "_translated.csv")
    df.to_csv(translated_path, index=False, encoding="utf-8")
    print(f"Wrote: {translated_path}")

    # Emotion analysis
    print(f"Emotion model: {cfg.emotion_model}")
    emo_pipe = build_emotion_pipe(cfg.emotion_model, dev)

    top_emotion, top_score, scores_json = analyze_emotions(
        emo_pipe,
        df["text_en"].tolist(),
        batch_size=cfg.emotion_batch_size,
    )

    df["top_emotion"] = top_emotion
    df["top_emotion_score"] = top_score
    df["emotion_scores_json"] = scores_json

    emotions_path = cfg.out_dir / (combined_path.stem + "_translated_emotions.csv")
    df.to_csv(emotions_path, index=False, encoding="utf-8")
    print(f"Wrote: {emotions_path}")

    # Wide export
    df_wide = expand_emotions_wide(df, json_col="emotion_scores_json")

    # URL-cleaned text for word clouds / analysis
    df_wide["text_clean"] = df_wide["text_en"].apply(remove_urls)

    wide_path = cfg.out_dir / (combined_path.stem + "_translated_emotions_wide.csv")
    df_wide.to_csv(wide_path, index=False, encoding="utf-8")
    print(f"Wrote: {wide_path}")

    # Optional plots/wordclouds
    if cfg.make_plots:
        plot_path = cfg.out_dir / "top_emotions_barh.png"
        plot_top_emotions(df_wide, out_path=plot_path)

    if cfg.make_wordclouds:
        wc_dir = cfg.out_dir / "wordclouds"
        make_wordclouds(df_wide, out_dir=wc_dir, top_k=6)

    print("Done.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Combine X datasets, translate FI->EN, run emotion analysis.")
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input CSV files (space-separated). Example: --input data/x_2023.csv data/x_2024.csv",
    )
    p.add_argument("--out-dir", type=Path, default=Path("out_x"), help="Output directory (default: out_x)")
    p.add_argument("--combined-name", type=str, default="x_combined.csv", help="Name of combined CSV")
    p.add_argument("--use-big-model", action="store_true", help="Use the larger FI->EN translation model")
    p.add_argument("--translation-batch-size", type=int, default=32)
    p.add_argument("--emotion-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512, help="HF translation max_length")
    p.add_argument("--max-chunk-chars", type=int, default=0, help="(reserved) not used in this script version")
    p.add_argument("--plots", action="store_true", help="Save a bar plot of top emotion counts")
    p.add_argument("--wordclouds", action="store_true", help="Save word clouds for top emotions (requires wordcloud)")

    a = p.parse_args()

    return Config(
        input_csvs=a.input,
        out_dir=a.out_dir,
        combined_name=a.combined_name,
        use_big_model=a.use_big_model,
        translation_batch_size=a.translation_batch_size,
        emotion_batch_size=a.emotion_batch_size,
        max_length=a.max_length,
        make_plots=a.plots,
        make_wordclouds=a.wordclouds,
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
