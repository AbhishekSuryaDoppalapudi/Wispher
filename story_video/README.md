# Story video: 3-5 minute animated video from your script

Pipeline: **script -> scenes -> clips (GPU) -> narration -> one mp4**.
Clips are made once and reused for every language; only the narration and subtitles are redone.

## 1. Write a script
Paragraphs separated by a blank line. Optionally add a `Visual:` line before a paragraph to tell
the video model what to draw (see `example_script.txt`). 3-5 minutes is about 450-750 words.

## 2. Plan the scenes
    python story_video/story_video.py plan myscript.txt --project mystory
Long paragraphs are split into scenes of about 22 words (5-8 s). Edit
`story_work/mystory/scenes.json` by hand to tweak any visual prompt.

## 3. Narration (English first)
    python story_video/story_video.py narrate --project mystory --lang English
Runs on CPU (about 1 min per scene on a laptop, so a 5 min video takes about 40-60 min).

## 4. Clips
- **Test on a laptop (placeholder colour clips):**
  `python story_video/story_video.py clips --project mystory`
- **Real clips (needs a big NVIDIA GPU, e.g. rented cloud A100/H100):** on that machine follow
  the setup in `gpu_clips.py`, then `python story_video/gpu_clips.py --project mystory`, and copy
  `story_work/mystory/clips/` back. `gpu_clips.py` is **untested** (no GPU on the dev laptop).
  A 5 min video is about 45-60 clips of about 5 s each.

## 5. Assemble
    python story_video/story_video.py assemble --project mystory --lang English
    python story_video/story_video.py assemble --project mystory --lang English --size 3840x2160 --burn

## 6. Other languages (reuses the same clips)
    python story_video/story_video.py narrate --project mystory --lang Telugu
    python story_video/story_video.py assemble --project mystory --lang Telugu

Output: `story_work/mystory/mystory_<Language>.mp4` plus an `.srt`.

## Notes
- LTX-2 weights are about 66 GiB and under the LTX Community License. Check it first.
- Clips shorter than the narration are slowed up to 1.5x, then the last frame is held.
- `Visual:` prompts work best when concrete (who, where, camera, light). Keep one visual idea per scene.
- Consistent characters across scenes are hard for any video model; describe them the same way each time.
