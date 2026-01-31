# report_workflow.py

## Overview
This script turns a PDF into structured, analyzable data.
It walks line by line through a document, listens for sentence boundaries,
translates Finnish into English, and then reads the emotional weather of each sentence.

The result is a small constellation of files:
plain text for the human eye, CSVs for analysis, and wide emotion tables
ready for statistics or visualization.

---

## What the script does (pipeline)

## 1. PDF extraction
- Uses ``PyMuPDF (fitz)`` to read the PDF page by page
- Keeps page numbers
- Extracts raw text

## 2. Text normalization
- Fixes hyphenation across line breaks  
  ``coro-\nnavirus → coronavirus``
- Collapses newlines and excess whitespace

## 3. Sentence splitting
- Splits on ``.``, ``!``, ``?``
- Protects common abbreviations (``e.g.``, ``Inc.``, months, etc.)
- Output unit: one sentence, with page number

## 4. Translation (FI → EN)
- Default model: ``Helsinki-NLP/opus-mt-fi-en``
- Long sentences are chunked (default ``400`` characters)
- Translation is batched (default ``32`` sentences per batch)
- Output: aligned Finnish–English sentence pairs

## 5. Emotion analysis
- Default model: ``SamLowe/roberta-base-go_emotions``
- Runs on English sentences
- Returns:
  - top emotion label
  - top emotion score
  - full emotion distribution (as JSON)

## 6. Emotion expansion (wide format)
- Expands the JSON emotion scores into columns
- One column per emotion label, prefixed with ``emo_``
- Missing values filled with ``0.0``


