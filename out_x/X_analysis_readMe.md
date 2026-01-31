# x_workflow.py

## Overview
This script processes X (Twitter) batch datasets end-to-end.
It combines multiple CSV files into a single dataset, translates post text
(from Finnish to English by default), performs emotion analysis using a
pretrained GoEmotions model, and optionally produces plots and word clouds.

The script is designed to be run from the command line and to scale from
small test batches to multi-year collections.

---

## What the script does

## 1. Combine datasets
- Reads multiple CSV files containing X posts
- Verifies a shared schema (expected columns)
- Concatenates rows into a single dataset
- Removes duplicate rows
- Writes a combined CSV

## 2. Text normalization
- Converts non-string values safely
- Collapses repeated whitespace
- Keeps original text intact in the combined dataset

## 3. Translation (FI → EN)
- Uses Hugging Face translation pipelines
- Default model: `Helsinki-NLP/opus-mt-fi-en`
- Optional large model: `Helsinki-NLP/opus-mt-tc-big-fi-en`
- Runs on GPU if available, otherwise CPU
- Translates in batches for efficiency

## 4. Emotion analysis
- Uses `SamLowe/roberta-base-go_emotions`
- Produces:
  - Top emotion label per post
  - Confidence score of the top emotion
  - Full emotion score distribution (JSON)

## 5. Wide emotion expansion
- Expands emotion score dictionaries into columns
- One column per emotion label, prefixed with `emo_`
- Missing values filled with `0.0`

## 6. Optional analysis outputs
- Horizontal bar plot of most frequent emotions
- Word clouds for the top emotions (based on cleaned English text)


---

## Notes and recommendations

### Performance
- For large datasets, reduce batch sizes if you encounter GPU memory errors
- CPU execution is slower but stable for small datasets

### Language assumptions
- Translation is Finnish → English by default
- If your text is already English, translation will still run unless you modify the script

### Interpretation of emotions
- Emotions are predicted per post, not per user or topic
- Scores indicate model confidence, not ground truth



