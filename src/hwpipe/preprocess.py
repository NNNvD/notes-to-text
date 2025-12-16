"""Image preprocessing for handwriting transcription."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from .config import get_settings


ROTATION_ANGLES = [0, 90, 180, 270]
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def detect_rotation(gray: np.ndarray) -> int:
    """Detect dominant orientation among 0/90/180/270 degrees."""

    angles = []
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(gray.shape) // 20)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            if angle > 180:
                angle -= 180
            if 80 <= angle <= 100:
                angles.append(90)
            elif 170 <= angle or angle <= 10:
                angles.append(0)
    if not angles:
        # Fallback using projection profile score
        scores = {}
        for angle in ROTATION_ANGLES:
            rotated = rotate_image(gray, angle)
            projection = rotated.sum(axis=1)
            scores[angle] = projection.std()
        return max(scores, key=scores.get)

    best_angle = max(set(angles), key=angles.count)
    return best_angle


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def apply_contrast_and_sharpen(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Light unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def preprocess_image(path: Path, output_dir: Path | None = None) -> Path:
    """Preprocess an image and save to normalized directory."""

    settings = get_settings()
    output_dir = output_dir or settings.normalized_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = detect_rotation(gray)
    rotated = rotate_image(image, angle)
    processed = apply_contrast_and_sharpen(rotated)

    normalized_path = output_dir / f"{Path(path).stem}.png"
    cv2.imwrite(str(normalized_path), processed)
    return normalized_path


def iter_images(folder: Path) -> Iterable[Path]:
    for ext in SUPPORTED_EXTS:
        yield from folder.glob(f"*{ext}")


def preprocess_folder(input_dir: Path, limit: int | None = None) -> list[Tuple[Path, Path]]:
    """Preprocess up to *limit* images from the input folder."""

    settings = get_settings()
    output_dir = settings.normalized_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_pairs: list[Tuple[Path, Path]] = []
    for idx, path in enumerate(sorted(iter_images(input_dir))):
        if limit is not None and idx >= limit:
            break
        normalized_path = preprocess_image(path, output_dir)
        processed_pairs.append((path, normalized_path))
    return processed_pairs
