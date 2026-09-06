# K0.3 — A From-Scratch GPT-Style Language Model

This repository contains the training and inference code for K0.3, a small decoder-only transformer language model trained entirely from scratch (no pretrained weights, no external tokenizer). The goal of this project was to understand and implement the full lifecycle of a language model — tokenizer training, pretraining, staged continual training, and evaluation — under the constraints of free/limited compute (Kaggle notebooks).

The model is a base language model: it completes text, it does not answer as an assistant. No instruction-tuning has been applied at this stage.

## Model architecture

- Decoder-only transformer (GPT-style), ~42.6M parameters
- 8 layers, 8 attention heads, 512-dimensional embeddings, 512-token context window
- Custom byte-level BPE tokenizer, 16,384 vocabulary size, trained from scratch on the training corpus
- Trained in PyTorch with mixed-precision (fp16), AdamW, cosine learning rate schedule with warmup

## Why staged training

A single Kaggle session has a limited GPU time budget, so the model could not be trained in one continuous run. Instead, training was split into stages: each stage adds a new batch of tokens to a growing corpus, and training resumes from the previous stage's checkpoint rather than starting over. To avoid the new data overwriting what the model had already learned, later stages use a lower ("warm-restart") learning rate and a reduced number of epochs relative to the total corpus size.

## Data sources

The training corpus is built from public, freely available English text:

- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — web articles and blog posts
- [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) — news articles from Common Crawl

Two additional sources — OpenSubtitles (dialogue) and BookCorpusOpen (novels) — were part of the original data plan to add conversational and narrative variety, but both became unavailable on Hugging Face partway through training (BookCorpusOpen's source data was taken down; the OpenSubtitles mirror's underlying file host stopped serving the archive). As a result, the final corpus is skewed toward web articles and news rather than the originally intended mix. This is discussed further under Limitations.

## Model versions and progress

All versions share the same architecture and tokenizer; only the amount and (attempted) composition of training data changes between them. Progress was tracked using a fixed set of test prompts across four categories — general knowledge, logical continuation, creative writing, and idiom completion — run with fixed sampling settings at each stage.

**K0.3.1 — ~200M tokens.** Baseline. Fluent grammar and sentence structure, but weak topical coherence — completions frequently drifted to unrelated subjects mid-paragraph. 0/4 idioms completed correctly.

**K0.3.2 — ~500M tokens.** Noticeable improvement in staying on-topic (e.g., a prompt about the UK stayed on UK/economy-related content instead of drifting to unrelated countries). Grammar errors present in K0.3.1 disappeared. 1/4 idioms completed correctly.

**K0.3.3 — ~800M tokens (final for this architecture).** Further gains in factual and structural detail — e.g., producing encyclopedia-style completions with plausible domain vocabulary (geography, statistics, named attribution in news-style text). However, idiom completion and logical reasoning did not improve over K0.3.2, which lines up with the missing dialogue/narrative data described above.

This model (K0.3.3) is the final version of the 42.6M-parameter architecture. Development is continuing with a larger, ~128M-parameter model.

## Files

- **`main.py`** — Initial training script. Trains the tokenizer if one doesn't exist yet, tokenizes the corpus into resumable shards, and trains the model from scratch with checkpointing, periodic evaluation, and early stopping.
- **`pull.py`** — Staged dataset preparation script. Streams data from multiple Hugging Face datasets without downloading everything at once, tracks how much of each source has already been used so re-running it for a new stage only fetches new documents, and merges + shuffles all data collected so far into a single training file.
- **`reeducate.py`** — Continual training script, used from the second stage onward. Loads a checkpoint produced by a previous stage, applies a reduced ("warm-restart") learning rate, and continues training on the enlarged cumulative dataset. Written for Kaggle notebooks, where the previous stage's checkpoint and the updated dataset are provided as input datasets.
- **`generate.py`** — Interactive inference script for plain text completion.
- **`test_generate.py`** — Same as `generate.py`, with an added logging mode: prompts and completions from a benchmark run can be saved, under a user-given title, to a text file for comparison across model versions.

## Known limitations

- The final corpus is skewed toward web articles and news; the planned dialogue and narrative sources were unavailable for most of training. This most likely explains the gap between the model's topical/factual improvements and its lack of progress on idiom recognition and logical reasoning.
- At 42.6M parameters, this model does not have reliable world knowledge. It produces fluent, grammatically plausible text that is often factually wrong — it is a language model, not a knowledge model.
- No instruction-tuning or RLHF has been applied. The model completes text; it does not follow instructions or answer questions in a conversational format.

## Next steps

A ~128M-parameter successor is in progress, with a training mix that adds curated logical and philosophical text (Socratic dialogue, thesis/antithesis structures) aimed specifically at improving reasoning ability, which was the clearest weak point of this model series.
