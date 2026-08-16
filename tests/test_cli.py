import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def test_infer_stdout_is_coordinate_only(tmp_path: Path):
    rng = np.random.default_rng(12)
    reference = rng.integers(0, 255, size=(100, 100), dtype=np.uint8)
    small = np.asarray(Image.fromarray(reference).resize((10, 10), Image.Resampling.BOX))
    search = rng.integers(0, 10, size=(100, 100), dtype=np.uint8)
    search[22:32, 31:41] = small
    ref_path, search_path = tmp_path / "ref.png", tmp_path / "search.png"
    Image.fromarray(reference).save(ref_path)
    Image.fromarray(search).save(search_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "infer.py", "--reference", str(ref_path), "--search", str(search_path), "--method", "baseline0"],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.strip().split()
    assert len(fields) == 2
    float(fields[0]), float(fields[1])


def test_infer_rejects_invalid_search_controls(tmp_path: Path):
    image = np.arange(10000, dtype=np.uint8).reshape(100, 100)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    for option, value in (("--top-k", "0"), ("--scale-range", "-0.1"), ("--rotation-range", "-1")):
        completed = subprocess.run(
            [sys.executable, "infer.py", "--reference", str(path), "--search", str(path), option, value],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert completed.stdout == ""
