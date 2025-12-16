"""Batch runner for handwriting transcription."""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from .config import Settings, get_settings
from .flags import flag_lines, write_flags
from .preprocess import SUPPORTED_EXTS, preprocess_image
from .transcribe import prompt_hash, transcribe_image


MANIFEST_HEADERS = [
    "filename",
    "sha256",
    "timestamp",
    "model",
    "prompt_hash",
    "status",
    "error_message",
]


def iter_inputs(folder: Path) -> Iterable[Path]:
    for ext in SUPPORTED_EXTS:
        yield from folder.glob(f"*{ext}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_row(manifest_path: Path, row: List[str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(MANIFEST_HEADERS)
        writer.writerow(row)


def process_file(path: Path, settings: Settings, args: argparse.Namespace, manifest_path: Path) -> None:
    raw_out = settings.out_raw_dir / f"{path.stem}.txt"
    flags_out = settings.out_flags_dir / f"{path.stem}.json"

    if raw_out.exists() and not args.force:
        write_manifest_row(
            manifest_path,
            [
                path.name,
                sha256_file(path),
                datetime.utcnow().isoformat(),
                args.model or settings.openai_model,
                prompt_hash(),
                "skipped",
                "exists",
            ],
        )
        return

    normalized_path = path
    if not args.no_preprocess:
        normalized_path = preprocess_image(path, settings.normalized_dir)

    status = "success"
    error_message = ""
    try:
        transcript = transcribe_image(normalized_path, model=args.model, api_key=settings.openai_api_key)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(transcript, encoding="utf-8")

        flags = flag_lines(transcript)
        write_flags(flags, flags_out)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = str(exc)

    write_manifest_row(
        manifest_path,
        [
            path.name,
            sha256_file(path),
            datetime.utcnow().isoformat(),
            args.model or settings.openai_model,
            prompt_hash(),
            status,
            error_message,
        ],
    )



def run_batch(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run handwriting transcription batch")
    parser.add_argument("--input", default=None, help="Input directory with images")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing step")

    parsed = parser.parse_args(args=args) if args is None or isinstance(args, list) else args

    settings = get_settings()
    settings.ensure_dirs()
    input_dir = Path(parsed.input) if parsed.input else settings.input_dir

    manifest_path = settings.manifest_dir / "run_manifest.csv"

    inputs = sorted(iter_inputs(input_dir))
    if parsed.limit is not None:
        inputs = inputs[: parsed.limit]

    for path in tqdm(inputs, desc="Processing", unit="file"):
        process_file(path, settings, parsed, manifest_path)


def main() -> None:
    run_batch()


if __name__ == "__main__":
    main()
