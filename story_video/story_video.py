"""Story video pipeline: script -> scenes -> narration -> clips -> one finished video.

Steps (run from the project root, e.g. `python story_video/story_video.py plan ...`):
  plan      split your script into scenes (scenes.json)
  narrate   make narration audio per scene in a language (English first, or any Indic language)
  clips     make placeholder clips so you can test everything on a laptop (no GPU needed)
            (real clips come from story_video/gpu_clips.py on a GPU machine)
  assemble  join clips + narration (+ subtitles) into one mp4

Clips are generated ONCE and reused for every language: translating only redoes narration
and subtitles, so a Telugu version costs minutes, not GPU hours.
"""
import argparse, hashlib, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WORK = os.path.join(ROOT, "story_work")
DEFAULT_STYLE = "Cinematic animated film, vibrant colors, smooth camera motion, high detail."


def proj(name):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    return d


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(cmd[:4]) + "\n" + r.stderr[-1500:])
    return r.stdout


def duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", path]).strip())


# ---------- plan ----------
def split_sentences(text, max_words):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()) or "story"


def plan(script_text, project, max_words=22, style=DEFAULT_STYLE):
    """Split a script into scenes and save scenes.json. Returns the scene list."""
    scenes = []
    for block in re.split(r"\n\s*\n", script_text.strip()):
        visual, lines = None, []
        for line in block.splitlines():
            if line.lower().startswith("visual:"):
                visual = line.split(":", 1)[1].strip()
            elif line.strip():
                lines.append(line.strip())
        text = " ".join(lines)
        if not text:
            continue
        for piece in split_sentences(text, int(max_words)):
            scenes.append({"id": len(scenes) + 1, "text": piece,
                           "visual": f"{style} {visual or piece}".strip()})
    d = proj(clean_name(project))
    json.dump(scenes, open(os.path.join(d, "scenes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return scenes


def cmd_plan(a):
    scenes = plan(open(a.script, encoding="utf-8").read(), a.project, a.max_words, a.style)
    words = sum(len(s["text"].split()) for s in scenes)
    print(f"{len(scenes)} scenes, {words} words (about {words / 150:.1f} min of narration).")


# ---------- narrate ----------
def narrate(project, lang="English", voice="Female", engine=None, cb=None):
    """Make narration audio per scene. `engine` is the module with translate/speak/trim_silence
    (the translator app); imported on demand when run from the command line."""
    if engine is None:
        import app as engine
    cb = cb or (lambda f, d: None)
    d = proj(clean_name(project))
    scenes = json.load(open(os.path.join(d, "scenes.json"), encoding="utf-8"))
    texts = [s["text"] for s in scenes]
    if lang != "English":
        cb(0.02, f"Translating {len(texts)} scenes to {lang}")
        texts = engine.translate(texts, "eng_Latn", engine.LANGS[lang])
    adir = os.path.join(d, "audio_" + lang)
    os.makedirs(adir, exist_ok=True)
    meta = []
    for k, (s, t) in enumerate(zip(scenes, texts)):
        # file name includes a hash of text+voice, so editing the script never reuses stale audio
        key = hashlib.md5(f"{t}|{voice}|{lang}".encode("utf-8")).hexdigest()[:8]
        raw, fin = f"{adir}/{s['id']:03d}_raw.wav", f"{adir}/{s['id']:03d}_{key}.wav"
        if not os.path.exists(fin):
            cb(0.1 + 0.9 * k / len(scenes), f"Voice {s['id']}/{len(scenes)}")
            engine.speak(t, voice, raw, lang)
            engine.trim_silence(raw, fin)
            os.remove(raw)
        meta.append({"id": s["id"], "text": t, "audio": fin, "dur": duration(fin)})
    json.dump(meta, open(os.path.join(d, f"narration_{lang}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return sum(m["dur"] for m in meta)


def cmd_narrate(a):
    secs = narrate(a.project, a.lang, a.voice, cb=lambda f, d: print(d))
    print(f"Narration done: {secs / 60:.1f} min")


# ---------- placeholder clips ----------
def make_placeholder_clips(project, size="1280x720"):
    d = proj(clean_name(project))
    scenes = json.load(open(os.path.join(d, "scenes.json"), encoding="utf-8"))
    cdir = os.path.join(d, "clips")
    os.makedirs(cdir, exist_ok=True)
    w, h = size.split("x")
    made = 0
    for s in scenes:
        out = os.path.join(cdir, f"{s['id']:03d}.mp4")
        if os.path.exists(out):
            continue
        hue = (s["id"] * 47) % 360
        c0, c1 = f"0x{(hue * 3) % 256:02x}4488", f"0x22{(hue * 5) % 256:02x}aa"
        sh(["ffmpeg", "-y", "-f", "lavfi", "-i",
            f"gradients=s={w}x{h}:d=5:r=24:speed=0.03:c0={c0}:c1={c1}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        made += 1
    return cdir, made


def cmd_clips(a):
    cdir, _ = make_placeholder_clips(a.project, a.size)
    print(f"Placeholder clips ready in {cdir} (replace with real ones from gpu_clips.py)")


# ---------- assemble ----------
def srt_time(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def assemble(project, lang="English", size="1920x1080", pad=0.3, burn=False, cb=None):
    """Join clips + narration into one video. Returns (video_path, srt_path, minutes)."""
    cb = cb or (lambda f, d: None)
    project = clean_name(project)
    d = proj(project)
    meta = json.load(open(os.path.join(d, f"narration_{lang}.json"), encoding="utf-8"))
    w, h = size.split("x")
    sdir = os.path.join(d, f"seg_{lang}")
    os.makedirs(sdir, exist_ok=True)
    srt_lines, t0, listing = [], 0.0, []
    for k, m in enumerate(meta):
        cb(k / len(meta), f"Scene {k + 1}/{len(meta)}")
        clip = os.path.join(d, "clips", f"{m['id']:03d}.mp4")
        if not os.path.exists(clip):
            raise FileNotFoundError(f"Missing clip {clip}. Make placeholder clips or add the real ones.")
        dur = m["dur"] + pad
        cd = duration(clip)
        # fit the clip to the narration: slow down up to 1.5x, then hold the last frame
        slow = min(max(dur / cd, 1.0), 1.5)
        vf = (f"setpts=PTS*{slow:.4f},tpad=stop_mode=clone:stop_duration={dur:.3f},"
              f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=24,"
              "format=yuv420p")
        seg = os.path.join(sdir, f"{m['id']:03d}.mp4")
        sh(["ffmpeg", "-y", "-i", clip, "-i", m["audio"], "-filter_complex",
            f"[0:v]{vf}[v];[1:a]apad[a]", "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-c:a", "aac", "-b:a", "192k",
            "-ar", "44100", seg])
        srt_lines.append(f"{m['id']}\n{srt_time(t0)} --> {srt_time(t0 + dur - 0.05)}\n{m['text']}\n")
        listing.append(f"file '{seg.replace(chr(92), '/')}'")
        t0 += dur
    lst = os.path.join(d, f"list_{lang}.txt")
    open(lst, "w", encoding="utf-8").write("\n".join(listing))
    srt = os.path.join(d, f"{project}_{lang}.srt")
    open(srt, "w", encoding="utf-8").write("\n".join(srt_lines))
    joined = os.path.join(d, f"joined_{lang}.mp4")
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    out = os.path.join(d, f"{project}_{lang}.mp4")
    if burn:  # burned subtitles (relative name + cwd avoids Windows path issues in filters)
        subprocess.run(["ffmpeg", "-y", "-i", joined, "-vf", f"subtitles={os.path.basename(srt)}",
                        "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-c:a", "copy", out],
                       cwd=d, check=True, capture_output=True)
    else:
        sh(["ffmpeg", "-y", "-i", joined, "-i", srt, "-c", "copy", "-c:s", "mov_text", out])
    return out, srt, t0 / 60


def cmd_assemble(a):
    out, _, mins = assemble(a.project, a.lang, a.size, a.pad, a.burn,
                            cb=lambda f, d: print(d))
    print(f"Done: {out}  ({mins:.1f} min)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("script"); s.add_argument("--project", required=True)
    s.add_argument("--max-words", type=int, default=22, help="words per scene (about 5-8 s)")
    s.add_argument("--style", default=DEFAULT_STYLE); s.set_defaults(f=cmd_plan)
    s = sub.add_parser("narrate"); s.add_argument("--project", required=True)
    s.add_argument("--lang", default="English"); s.add_argument("--voice", default="Female")
    s.set_defaults(f=cmd_narrate)
    s = sub.add_parser("clips"); s.add_argument("--project", required=True)
    s.add_argument("--size", default="1280x720"); s.set_defaults(f=cmd_clips)
    s = sub.add_parser("assemble"); s.add_argument("--project", required=True)
    s.add_argument("--lang", default="English"); s.add_argument("--size", default="1920x1080")
    s.add_argument("--pad", type=float, default=0.3, help="seconds of pause after each scene")
    s.add_argument("--burn", action="store_true", help="burn subtitles into the picture")
    s.set_defaults(f=cmd_assemble)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
