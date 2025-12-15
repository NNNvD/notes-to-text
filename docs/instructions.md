Yes—given **108 pages**, **no privacy constraints**, and **plain text output**, the fastest and cleanest setup is:

* **Run locally** (your machine) for speed and simplicity
* Use **GitHub** for versioned code + outputs + review trace
* Use **Codex** to generate the repo and scripts end-to-end

Below is a Codex-ready implementation brief (you can paste it directly).

---

# Codex brief: Handwriting transcription pipeline (Option A)

## Goal

Create a repo that converts a folder of page images (108 pages) into **plain text transcripts** using a vision-capable model, with minimal preprocessing and lightweight review flagging.

## Non-goals

* No form extraction
* No structured output (no bounding boxes)
* No database, no UI beyond simple files

## Repo name

`handwriting-transcribe-pipeline`

## Tech stack

* Python 3.11+
* `openai` SDK
* `opencv-python` (or Pillow if you prefer, but OpenCV makes rotation easier)
* `pydantic` (optional, for clean configs)
* `tqdm` for progress

## Folder structure

```
handwriting-transcribe-pipeline/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  data/
    in/            # user drops images here
    normalized/    # pipeline writes normalized images here
    out_raw/       # raw model transcripts (.txt)
    out_flags/     # review flags (.json)
    out_final/     # corrected transcripts (.txt) - manual edits
    manifests/     # CSV logs
  src/
    hwpipe/
      __init__.py
      config.py
      preprocess.py
      transcribe.py
      flags.py
      batch.py
  scripts/
    run_batch.py
```

## Input assumptions

* Mixed PNG/JPG
* Some pages may be sideways (like the sample)
* One page per image file

## Output requirements

For each input image `X.jpg`:

* `data/out_raw/X.txt` (verbatim transcript)
* `data/out_flags/X.json` (line numbers to review + reasons)
* Optional: `data/normalized/X.png` (normalized image)
* After manual edits: `data/out_final/X.txt`

## Preprocessing (minimal, fast)

Implement `preprocess_image()`:

1. **Auto-rotate**: detect dominant text orientation; if sideways, rotate 90/180 as needed.

   * Heuristic acceptable: use Hough lines / projection profile to pick rotation among {0,90,180,270}.
2. **Perspective**: skip unless obvious; keep it simple for speed.
3. **Contrast**: apply mild CLAHE on luminance or grayscale normalization.
4. **Sharpen**: light unsharp mask.
5. Save normalized as PNG.

Explicitly avoid binarization and aggressive denoising.

## Transcription prompt (must be strict)

Use a single prompt template, e.g.:

**System (or leading instruction):**

* “You are a transcription engine. Transcribe handwritten text verbatim.”

**Rules:**

* Preserve line breaks as seen.
* Do not correct spelling or grammar.
* If uncertain, write `[unclear]` (do not guess).
* If a token is partly legible, write the legible part + `[unclear]` (e.g., `psycho[unclear]`).
* Output only the transcription, no commentary.

## Transcription implementation

Implement `transcribe_image(path_normalized) -> str`:

* Call the vision-capable model with image input.
* Return plain text only.
* Retry up to 2 times on transient errors with exponential backoff.
* Deterministic-ish: set temperature low (if supported) to reduce variation.
* Store the exact prompt and model name in manifest.

## Review flagging heuristics

Implement `flag_lines(text: str) -> dict` returning:

* `lines`: list of `{line_no, reason}`
  Flag if:
* line contains `[unclear]`
* line contains long weird tokens (e.g., >25 chars without spaces)
* line has high non-alphabetic ratio
* (Optional) contains “?” or suspicious punctuation density

Write JSON to `data/out_flags/X.json`.

## Batch runner

`scripts/run_batch.py` should:

* Iterate `data/in/*.{png,jpg,jpeg,webp}`
* For each:

  * preprocess → `data/normalized/`
  * transcribe → `data/out_raw/`
  * flag → `data/out_flags/`
* **Resumable**: skip if `out_raw` exists unless `--force`
* Write `data/manifests/run_manifest.csv` with:

  * filename, sha256, timestamps, model, prompt_hash, status, error_message
* CLI args:

  * `--input data/in`
  * `--force`
  * `--limit 10`
  * `--model <name>`
  * `--no-preprocess` (optional)

## README usage

Provide:

1. Install:

   * `python -m venv .venv`
   * activate
   * `pip install -e .`
2. Configure:

   * copy `.env.example` → `.env`
   * set `OPENAI_API_KEY=...`
3. Run:

   * `python scripts/run_batch.py`
4. Review workflow:

   * Open `out_raw/X.txt` and `out_flags/X.json`
   * Copy to `out_final/X.txt` and correct only flagged lines
5. Completion check:

   * ensure every input has `out_final` (or accept raw if no flags)

## Acceptance tests (must implement)

* Running on a folder with 3 sample images creates all expected outputs.
* Re-running without `--force` does not reprocess already completed files.
* Manifest CSV is created and updated.
* Flag JSON line numbers match the raw transcript line splitting.

## .gitignore

Ignore:

* `data/in/*` (optional; but for 108 pages you may want to keep them out of git)
* `data/normalized/*`
* any `.env`
  Keep:
* `data/out_raw` and `data/out_final` (your choice; I recommend keeping `out_final` in git, `out_raw` optional)
* `manifests`

---

# Operational recommendation for your 108 pages

* Do **one dry run** on 5 pages to confirm orientation handling and prompt behavior.
* Then run the full batch.
* Review only flagged lines; for this handwriting, expect roughly **1–2 minutes per page** for verification once the pipeline is stable.

---

If you want, I can also provide a **ready-to-paste GitHub Issues checklist** (setup, implementation tasks, and QA steps) that you can assign to Codex verbatim.
