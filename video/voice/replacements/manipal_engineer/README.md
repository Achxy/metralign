# Sentence replacement audio

Place a replacement at `<segment_id>.wav` using the stable IDs in
`video/film.yaml`. A replacement takes precedence over Mimika output; then
`make video-voice` remeasures it with ffprobe and regenerates the authoritative
timeline and captions. Animation code does not need to change.

Use clean mono or stereo PCM/WAV whenever possible. Final assembly resamples
the complete narration stem to the sample-rate and loudness policy declared in
`film.yaml`; speech is not time-compressed to preserve an old edit.
