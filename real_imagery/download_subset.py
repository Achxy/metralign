#!/usr/bin/env python3
"""Download and verify the pinned, minimal real-SEM subset."""

from __future__ import annotations

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources.json"


def file_digest(path: Path, algorithm: str) -> str:
    digest = md5(usedforsecurity=False) if algorithm == "md5" else sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_files(manifest: dict) -> list[tuple[dict, dict, dict]]:
    rows = []
    for dataset in manifest["datasets"]:
        for area in dataset["areas"]:
            for record in area["files"]:
                rows.append((dataset, area, record))
    return rows


def verify_existing(path: Path, record: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(record["size"])
        and file_digest(path, "md5") == record["md5"]
    )


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "metralign-real-imagery/1"})
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        raise FileExistsError(f"partial download already exists: {part}")
    try:
        with urlopen(request, timeout=120) as response, part.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        part.replace(destination)
    except Exception:
        if part.exists():
            part.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(SOURCES.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, area, record in expected_files(manifest):
        directory = args.output_dir / dataset["id"] / area["area"]
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / record["name"]
        if destination.exists() and not verify_existing(destination, record):
            raise ValueError(f"existing file fails pinned size/MD5: {destination}")
        if not destination.exists():
            download(record["url"], destination)
        if not verify_existing(destination, record):
            raise ValueError(f"download fails pinned size/MD5: {destination}")
        records.append(
            {
                "dataset": dataset["id"],
                "area": area["area"],
                "magnification_k": record["magnification_k"],
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "md5": record["md5"],
                "sha256": file_digest(destination, "sha256"),
            }
        )
        print(f"verified {destination}")

    output = {
        "schema_version": 1,
        "source_manifest": str(SOURCES),
        "source_manifest_sha256": sha256(SOURCES.read_bytes()).hexdigest(),
        "files": records,
    }
    record_path = args.output_dir / "download-record.json"
    record_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
