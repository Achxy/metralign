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

The mathematical scenes use Manim's semantic `MathTex` pipeline. Install a
native LaTeX distribution with `latex` and `dvisvgm`; on Apple Silicon the
tested setup is:

```bash
brew install texlive dvisvgm
```

The renderer discovers the Homebrew TeX tree automatically and keeps Manim's
SVG term identifiers intact for `TransformMatchingTex`. PDF-to-SVG
compatibility wrappers are intentionally unsupported because they discard
those identifiers.

`make video-preview` and `make video` run the same evidence → draft
resolution → Mimika sentence audio → strict resolution → Manim → FFmpeg
pipeline. Replacing a sentence WAV under
`video/voice/replacements/manipal_engineer/` changes timing without changing
scene code.

The main output is `video/final/Metralign_Demo.mp4`. No generated scientific
imagery is used. Every technical frame is built from project evidence or a
source-bound external asset.

[`VISUAL_LANGUAGE.md`](VISUAL_LANGUAGE.md) defines the film's layout,
typography, evidence, and motion rules. It borrows the explanatory grammar of
mathematical animation—continuity, progressive disclosure, and semantic
transforms—without imitating another studio's branding.
