VIDEO_PYTHON := video/.venv/bin/python

.PHONY: video-evidence video-voice video-scenes video-preview video video-qa

video-evidence:
	$(VIDEO_PYTHON) video/tools/build.py evidence

video-voice:
	$(VIDEO_PYTHON) video/tools/build.py voice

video-scenes:
	$(VIDEO_PYTHON) video/tools/build.py scenes --profile preview

video-preview:
	$(VIDEO_PYTHON) video/tools/build.py preview

video:
	$(VIDEO_PYTHON) video/tools/build.py final

video-qa:
	$(VIDEO_PYTHON) video/tools/build.py qa
