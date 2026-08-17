# Metralign technical film

`film.yaml` is the only manually edited production specification. Evidence,
narration audio, captions, timing, scene manifests, plots, and final encodes are
derived from it and the checked-in project records.

The public film name is **Metralign**. The older Drift-Sense wording is retained
only as submission context; LatticeLock is not a current project name.

Build interfaces:

```bash
make video-preview
make video-voice
make video-scenes
make video
make video-qa
```

Create the isolated renderer environment once:

```bash
python3.12 -m venv video/.venv
video/.venv/bin/python -m pip install -r video/requirements.txt
```

`make video-preview` and `make video` run the same evidence → draft
resolution → Mimika sentence audio → strict resolution → Manim → FFmpeg
pipeline. Replacing a sentence WAV under
`video/voice/replacements/manipal_engineer/` changes timing without changing
scene code.

The main output is `video/final/Metralign_Demo.mp4`. No generated scientific
imagery is used. Every technical frame is built from project evidence or a
source-bound external asset.
