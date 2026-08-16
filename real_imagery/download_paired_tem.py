#!/usr/bin/env python3
"""Download and verify the pinned registered MiniTEM archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from real_imagery.protocol import digest_file


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "paired_tem_source.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(SOURCE.read_text())
    record = manifest["archive"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / record["name"]
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if (
            destination.stat().st_size != int(record["bytes"])
            or digest_file(destination, "md5") != record["md5"]
        ):
            raise ValueError(f"existing archive fails pinned size/MD5: {destination}")
    else:
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "metralign-real-imagery/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(record["url"], headers=headers)
        try:
            response = urlopen(request, timeout=120)
            if offset:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                    response.close()
                    raise ValueError("server did not honor the requested resume byte range")
            mode = "ab" if offset else "xb"
            with response, part.open(mode) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if part.stat().st_size != int(record["bytes"]):
                raise ValueError(
                    f"partial archive has {part.stat().st_size} bytes; expected {record['bytes']}; rerun to resume"
                )
            part.replace(destination)
        except Exception:
            raise
        if (
            destination.stat().st_size != int(record["bytes"])
            or digest_file(destination, "md5") != record["md5"]
        ):
            raise ValueError(f"download fails pinned size/MD5: {destination}")
    verification = {
        "schema_version": 1,
        "source_manifest_sha256": digest_file(SOURCE),
        "archive": {
            "path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "md5": record["md5"],
            "sha256": digest_file(destination),
        },
    }
    record_path = args.output_dir / "paired-tem-download-record.json"
    record_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
