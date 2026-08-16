"""Pure helpers for the real-imagery evaluation protocol."""

from __future__ import annotations

from hashlib import md5, sha256
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DIGITAL_POSITIONS = (
    (0.22, 0.22),
    (0.78, 0.22),
    (0.50, 0.50),
    (0.22, 0.78),
    (0.78, 0.78),
)


def crop_geometry(
    shape: tuple[int, int],
    position: tuple[float, float],
    nominal_scale: float,
    minimum_crop_size: int,
) -> tuple[int, int, int, int, tuple[float, float]]:
    height, width = shape
    if minimum_crop_size < 8:
        raise ValueError("minimum_crop_size must be at least 8 pixels")
    crop_width = max(minimum_crop_size, int(round(width * nominal_scale)))
    crop_height = max(minimum_crop_size, int(round(height * nominal_scale)))
    if crop_width > width or crop_height > height:
        raise ValueError("crop does not fit within the source image")
    target_x = float(position[0]) * (width - 1)
    target_y = float(position[1]) * (height - 1)
    x0 = int(round(target_x - (crop_width - 1) / 2.0))
    y0 = int(round(target_y - (crop_height - 1) / 2.0))
    x0 = min(max(x0, 0), width - crop_width)
    y0 = min(max(y0, 0), height - crop_height)
    ground_truth = (
        x0 + (crop_width - 1) / 2.0,
        y0 + (crop_height - 1) / 2.0,
    )
    return x0, y0, crop_width, crop_height, ground_truth


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = md5(usedforsecurity=False) if algorithm == "md5" else sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sem_content(path: Path, crop_spec: dict) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    expected = (
        int(crop_spec["expected_height_px"]),
        int(crop_spec["expected_width_px"]),
    )
    if gray.shape != expected:
        raise ValueError(f"unexpected source dimensions for {path}: {gray.shape} != {expected}")
    rows = int(crop_spec["caption_rows_removed_from_bottom"])
    if rows <= 0 or rows >= gray.shape[0]:
        raise ValueError(f"invalid caption crop: {rows}")
    return np.ascontiguousarray(gray[:-rows])


def digital_crop_pair(
    search: np.ndarray,
    position: tuple[float, float],
    nominal_scale: float = 0.1,
    minimum_crop_size: int = 32,
) -> tuple[np.ndarray, tuple[float, float], dict]:
    """Build an exact digital reference from real acquired pixels.

    The reference is an interpolation of a crop from ``search``.  The returned
    coordinate is therefore digital construction truth, not microscope truth.
    """
    height, width = search.shape
    x0, y0, crop_width, crop_height, ground_truth = crop_geometry(
        search.shape, position, nominal_scale, minimum_crop_size
    )
    crop = search[y0 : y0 + crop_height, x0 : x0 + crop_width]
    reference = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)
    details = {
        "position_fraction": [float(position[0]), float(position[1])],
        "crop_box_xywh": [x0, y0, crop_width, crop_height],
        "minimum_crop_size_px": minimum_crop_size,
        "reference_resampler": "OpenCV INTER_LANCZOS4",
    }
    return np.ascontiguousarray(reference), ground_truth, details


def informative_crop_positions(
    image: np.ndarray,
    *,
    count: int = 5,
    nominal_scale: float = 0.1,
    minimum_crop_size: int = 8,
) -> list[dict]:
    """Select deterministic, non-overlapping GT crops without using predictions.

    Candidates are a fixed 9x9 interior grid. Informativeness is mean absolute
    residual after a Gaussian blur (sigma 3 px), evaluated only on the sharp GT
    crop. Ties are resolved by row then column. This rejects blank padding while
    remaining independent of every localization output and Low frame.
    """
    fractions = np.linspace(0.12, 0.88, 9)
    candidates = []
    for y_fraction in fractions:
        for x_fraction in fractions:
            position = (float(x_fraction), float(y_fraction))
            x0, y0, width, height, center = crop_geometry(
                image.shape, position, nominal_scale, minimum_crop_size
            )
            crop = np.asarray(image[y0 : y0 + height, x0 : x0 + width], dtype=np.float32)
            residual = crop - cv2.GaussianBlur(crop, (0, 0), 3.0)
            candidates.append(
                {
                    "position_fraction": [position[0], position[1]],
                    "crop_box_xywh": [x0, y0, width, height],
                    "center_xy": [float(center[0]), float(center[1])],
                    "highpass_mean_absolute": float(np.mean(np.abs(residual))),
                }
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["highpass_mean_absolute"],
            item["position_fraction"][1],
            item["position_fraction"][0],
        ),
    )
    selected = []
    for candidate in ranked:
        x0, y0, width, height = candidate["crop_box_xywh"]
        if all(
            x0 + width <= prior["crop_box_xywh"][0]
            or prior["crop_box_xywh"][0] + prior["crop_box_xywh"][2] <= x0
            or y0 + height <= prior["crop_box_xywh"][1]
            or prior["crop_box_xywh"][1] + prior["crop_box_xywh"][3] <= y0
            for prior in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)} non-overlapping informative crops")
    for rank, item in enumerate(selected):
        item["rank"] = rank
    return selected


def feature_registration_proxy(
    reference: np.ndarray,
    search: np.ndarray,
    nominal_scale: float,
) -> dict:
    """Estimate an independent affine correspondence and expose its quality gate."""
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    keypoints_reference, descriptors_reference = sift.detectAndCompute(reference, None)
    keypoints_search, descriptors_search = sift.detectAndCompute(search, None)
    base = {
        "method": "SIFT + Lowe ratio 0.72 + RANSAC partial affine",
        "reference_keypoints": len(keypoints_reference),
        "search_keypoints": len(keypoints_search),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_inlier_reprojection_px": None,
        "estimated_scale": None,
        "estimated_rotation_deg": None,
        "reference_center_in_search_px": None,
        "accepted_for_agreement": False,
        "rejection_reasons": [],
    }
    if descriptors_reference is None or descriptors_search is None:
        base["rejection_reasons"] = ["missing descriptors"]
        return base
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors_reference, descriptors_search, k=2
    )
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    base["good_matches"] = len(good)
    if len(good) < 4:
        base["rejection_reasons"] = ["fewer than four ratio-test matches"]
        return base
    source = np.float32([keypoints_reference[item.queryIdx].pt for item in good])
    destination = np.float32([keypoints_search[item.trainIdx].pt for item in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=50,
    )
    if matrix is None or mask is None:
        base["rejection_reasons"] = ["RANSAC did not return an affine estimate"]
        return base
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / len(good)
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(predicted - destination, axis=1)
    median_residual = float(np.median(residuals[inlier_mask])) if inliers else math.inf
    linear = matrix[:, :2]
    scale = float(math.sqrt(abs(np.linalg.det(linear))))
    rotation = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    center = np.array([(reference.shape[1] - 1) / 2.0, (reference.shape[0] - 1) / 2.0])
    mapped = linear @ center + matrix[:, 2]
    reasons = []
    if len(good) < 16:
        reasons.append("fewer than 16 ratio-test matches")
    if inliers < 12:
        reasons.append("fewer than 12 RANSAC inliers")
    if ratio < 0.25:
        reasons.append("inlier ratio below 0.25")
    if median_residual > 3.0:
        reasons.append("median inlier reprojection above 3 px")
    if not 0.65 * nominal_scale <= scale <= 1.35 * nominal_scale:
        reasons.append("estimated scale outside ±35% of nominal magnification ratio")
    if not (0.0 <= mapped[0] < search.shape[1] and 0.0 <= mapped[1] < search.shape[0]):
        reasons.append("mapped reference center outside search image")
    base.update(
        {
            "inliers": inliers,
            "inlier_ratio": float(ratio),
            "median_inlier_reprojection_px": median_residual,
            "estimated_scale": scale,
            "estimated_rotation_deg": rotation,
            "reference_center_in_search_px": [float(mapped[0]), float(mapped[1])],
            "accepted_for_agreement": not reasons,
            "rejection_reasons": reasons,
        }
    )
    return base


def error_metrics(values: list[float], *, thresholds: tuple[float, ...]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    output = {
        "count": int(array.size),
        "mean_px": float(np.mean(array)),
        "median_px": float(np.median(array)),
        "p95_px": float(np.percentile(array, 95, method="linear")),
        "maximum_px": float(np.max(array)),
    }
    for threshold in thresholds:
        label = str(threshold).replace(".", "_")
        output[f"count_le_{label}px"] = int(np.count_nonzero(array <= threshold))
        output[f"fraction_le_{label}px"] = float(np.mean(array <= threshold))
    return output
