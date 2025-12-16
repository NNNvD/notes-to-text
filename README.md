# Handwriting Transcription Pipeline

Batch pipeline for preprocessing handwritten page images, running an OpenAI vision model for strict verbatim transcription, and flagging lines that need review.

## Setup

1. Python 3.11+
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. Configure environment:

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY and optional OPENAI_MODEL
```

## Running the pipeline

Place input images (PNG/JPG/WebP) in `data/in/` (the repository also includes a dry-run batch under `images/` you can point `--input` to).

```bash
python scripts/run_batch.py \
  --input data/in \
  --model gpt-4o-mini \
  --limit 5
```

Key flags:

- `--force`: reprocess even if outputs already exist
- `--limit N`: process only N images
- `--model NAME`: override model name
- `--no-preprocess`: skip normalization step

Outputs for each `X.jpg`:

- `data/normalized/X.png` — rotated/contrast-enhanced image
- `data/out_raw/X.txt` — raw transcript (strict `[unclear]` policy)
- `data/out_flags/X.json` — review hints with line numbers
- `data/out_final/X.txt` — manual corrections (user maintained)
- `data/manifests/run_manifest.csv` — run log (filename, hash, prompt hash, status)

## Review workflow

1. Inspect `data/out_flags/*.json` to identify lines needing attention.
2. Open matching `data/out_raw/*.txt`, copy to `data/out_final/*.txt` and correct only flagged lines.
3. Re-run with `--force` if you want to regenerate outputs after changes.

## Notes

- Preprocessing rotates pages heuristically and applies light CLAHE + unsharp mask; aggressive denoising and binarization are intentionally avoided.
- The transcription prompt enforces verbatim output and `[unclear]` tokens when confidence is low.
- Manifest rows are appended on each run; reruns without `--force` skip completed files but still log a `skipped` status.
