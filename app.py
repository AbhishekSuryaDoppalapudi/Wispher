import os, gc, re, sys, json, time, shutil, subprocess, tempfile
import torch, numpy as np, soundfile as sf
import whisper
import gradio as gr
import access_log


def _patch_gradio_file_retry():
    """Windows security scanners briefly lock freshly uploaded big videos, so the preview fails with
    'PermissionError' (shown as 'Error' in the video box). Retry opening for a few seconds instead."""
    import asyncio, types
    import gradio.ranged_response as rr

    real = rr.aiofiles

    class _RetryOpen:
        def __init__(self, *a, **k):
            self.a, self.k, self.cm = a, k, None

        async def __aenter__(self):
            for i in range(20):
                try:
                    self.cm = real.open(*self.a, **self.k)
                    return await self.cm.__aenter__()
                except PermissionError:
                    if i == 19:
                        raise
                    await asyncio.sleep(0.5)

        async def __aexit__(self, *exc):
            return await self.cm.__aexit__(*exc)

    rr.aiofiles = types.SimpleNamespace(open=_RetryOpen)

    # Starlette's own file route (used for video previews) needs the same retry
    import anyio
    import starlette.responses as sr
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _open_retry(self):
        for i in range(40):
            try:
                file = await anyio.open_file(self.path, mode="rb")
                break
            except PermissionError:
                if i == 39:
                    raise
                await asyncio.sleep(0.5)
        try:
            yield file
        finally:
            with anyio.CancelScope(shield=True):
                await file.aclose()

    sr.FileResponse._open_file = _open_retry


try:
    _patch_gradio_file_retry()
except Exception as _e:  # never block the app over this
    print("note: file-retry patch not applied:", _e)
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    from IndicTransToolkit import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All 22 scheduled Indian languages + English (IndicTrans2 / FLORES codes)
LANGS = {
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", "Dogri": "doi_Deva",
    "English": "eng_Latn", "Gujarati": "guj_Gujr", "Hindi": "hin_Deva", "Kannada": "kan_Knda",
    "Kashmiri": "kas_Arab", "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Malayalam": "mal_Mlym",
    "Manipuri": "mni_Beng", "Marathi": "mar_Deva", "Nepali": "npi_Deva", "Odia": "ory_Orya",
    "Punjabi": "pan_Guru", "Sanskrit": "san_Deva", "Santali": "sat_Olck", "Sindhi": "snd_Arab",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab",
}
CODE_TO_NAME = {v: k for k, v in LANGS.items()}

# Languages Indic Parler-TTS can speak (conservative list - check the model card before adding
# more). Others still get translated subtitles.
TTS_LANGS = {"Assamese", "Bengali", "Bodo", "Dogri", "English", "Gujarati", "Hindi", "Kannada",
             "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", "Nepali", "Odia",
             "Punjabi", "Sanskrit", "Santali", "Tamil", "Telugu"}

# Spoken languages Whisper can recognise in the uploaded video
WHISPER_TO_FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ur": "urd_Arab", "as": "asm_Beng", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
}
FLORES_TO_WHISPER = {v: k for k, v in WHISPER_TO_FLORES.items()}

# Indic Parler-TTS has named speakers per language; using the same name for every line keeps
# the voice consistent (names from the model card). (female, male). Languages without their own
# speaker (Konkani, Maithili, Santali) use the Hindi voices.
SPEAKERS = {
    "Assamese": ("Sita", "Amit"), "Bengali": ("Aditi", "Arjun"), "Bodo": ("Maya", "Bikram"),
    "Dogri": ("Karan", "Karan"), "English": ("Mary", "Thoma"), "Gujarati": ("Neha", "Yash"),
    "Hindi": ("Divya", "Rohit"), "Kannada": ("Anu", "Suresh"), "Malayalam": ("Anjali", "Harish"),
    "Manipuri": ("Laishram", "Laishram"), "Marathi": ("Sunita", "Sanjay"),
    "Nepali": ("Amrita", "Amrita"), "Odia": ("Debjani", "Manas"),
    "Punjabi": ("Divjot", "Gurpreet"), "Sanskrit": ("Aryan", "Aryan"),
    "Tamil": ("Jaya", "Jaya"), "Telugu": ("Lalitha", "Prakash"),
}
VOICES = ["Female", "Male"]


def voice_description(lang, voice):
    f, m = SPEAKERS.get(lang, SPEAKERS["Hindi"])
    name = f if voice == "Female" else m
    return (f"{name}'s voice is monotone yet slightly fast in delivery, with a very clear audio "
            "and a very close recording that almost has no background noise.")


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def purge_old_runs(max_age_sec=3600):
    """Delete leftover work folders from earlier runs."""
    root = tempfile.gettempdir()
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if d.startswith("ivt_") and os.path.isdir(full) and time.time() - os.path.getmtime(full) > max_age_sec:
            shutil.rmtree(full, ignore_errors=True)


def free():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------- 1. Speech to text (with timestamps) ----------
def transcribe(audio_path, src_choice):
    model = whisper.load_model("turbo", device=DEVICE)
    lang = None if src_choice == "Auto-detect" else FLORES_TO_WHISPER[LANGS[src_choice]]
    result = model.transcribe(audio_path, language=lang, task="transcribe",
                              fp16=(DEVICE == "cuda"))
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result["segments"] if s["text"].strip()]
    detected = result.get("language") or lang
    del model
    free()
    return segs, detected


# ---------- 2. Translation (IndicTrans2) ----------
def pick_model(src, tgt):
    if src == "eng_Latn":
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    if tgt == "eng_Latn":
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    return "ai4bharat/indictrans2-indic-indic-dist-320M"


def translate(texts, src, tgt, batch_size=16):
    if src == tgt:
        return list(texts)
    name = pick_model(src, tgt)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(DEVICE).eval()
    ip = IndicProcessor(inference=True)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = ip.preprocess_batch(texts[i:i + batch_size], src_lang=src, tgt_lang=tgt)
        enc = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                  return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, use_cache=False, min_length=0, max_length=256,
                                 num_beams=4, num_return_sequences=1)
        dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out += ip.postprocess_batch(dec, lang=tgt)
    del model
    free()
    return out


# ---------- 3. Subtitles ----------
def ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, t) in enumerate(zip(segs, texts), 1):
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{t}\n\n")


# ---------- 4. Dubbed voice (Indic Parler-TTS) ----------
_tts = {}


def load_tts():
    if not _tts:
        name = "ai4bharat/indic-parler-tts"
        from parler_tts import ParlerTTSForConditionalGeneration  # lazy: needs torchaudio
        m = ParlerTTSForConditionalGeneration.from_pretrained(name).to(DEVICE)
        _tts["model"] = m
        _tts["tok"] = AutoTokenizer.from_pretrained(name)
        _tts["desc_tok"] = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
    return _tts


def speak(text, voice, out_path, lang="Hindi"):
    t = load_tts()
    d = t["desc_tok"](voice_description(lang, voice), return_tensors="pt").to(DEVICE)
    p = t["tok"](text, return_tensors="pt").to(DEVICE)
    torch.manual_seed(42)  # keeps the voice consistent across lines
    with torch.no_grad():
        g = t["model"].generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                prompt_input_ids=p.input_ids,
                                prompt_attention_mask=p.attention_mask)
    audio = g.cpu().numpy().squeeze().astype(np.float32)
    sf.write(out_path, audio, t["model"].config.sampling_rate)


def trim_silence(wav_path, out_path, thresh=-45):
    """Cut the silence TTS leaves at the start/end so the line starts exactly when we say."""
    a = AudioSegment.from_wav(wav_path)
    lead = detect_leading_silence(a, silence_threshold=thresh)
    trail = detect_leading_silence(a.reverse(), silence_threshold=thresh)
    a[lead:len(a) - trail].export(out_path, format="wav")


def fit_to_slot(wav_path, slot_sec, out_path, max_speed=1.6, min_speed=0.85):
    """Make a line last about as long as the original speaker's line (speed it up if too
    long, slow it slightly if too short) so voice and mouth movement line up."""
    dur = sf.info(wav_path).duration
    speed = 1.0
    if slot_sec > 0.3:
        speed = min(max(dur / slot_sec, min_speed), max_speed)
    run(["ffmpeg", "-y", "-i", wav_path, "-filter:a", f"atempo={speed:.3f}",
         "-ar", "44100", "-ac", "1", out_path])


def build_dub_track(segs, texts, voice, total_ms, tmp, progress, lang="Hindi"):
    track = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    n = len(segs)
    busy_until = 0  # ms; a line never starts before the previous one has finished
    for i, (s, txt) in enumerate(zip(segs, texts)):
        progress(0.55 + 0.3 * i / n, desc=f"Generating voice {i + 1}/{n}")
        if not txt.strip():
            continue
        raw, trim, fit = f"{tmp}/seg{i}.wav", f"{tmp}/trim{i}.wav", f"{tmp}/fit{i}.wav"
        speak(txt, voice, raw, lang)
        trim_silence(raw, trim)
        # Aim for the speaker's own mouth-movement time; allow a little spill into the gap
        nxt = segs[i + 1]["start"] if i + 1 < n else s["end"] + 1.0
        slot_end = max(s["end"], min(nxt - 0.1, s["end"] + 0.4))
        fit_to_slot(trim, slot_end - s["start"], fit)
        clip = AudioSegment.from_wav(fit)
        pos = max(int(s["start"] * 1000), busy_until)
        track = track.overlay(clip, position=pos)
        busy_until = pos + len(clip) + 80
    return track


# ---------- Full pipeline ----------
def video_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                       capture_output=True, text=True)
    w, h = r.stdout.strip().split("x")[:2]
    return int(w), int(h)


def process(video, src_choice, tgt_name, voice, do_dub, keep_bg, burn,
            clip_start=0, clip_len=0, res="Original resolution", token=None, progress=gr.Progress()):
    """Checks the email verification, runs one translation and records it in the Excel log."""
    if not video:
        raise gr.Error("Upload a video first.")
    if not has_audio(video):
        raise gr.Error("This video has no audio track to translate.")
    person = access_log.consume(token)
    if person is None:
        raise gr.Error("Verify your email first: fill in 'Who is translating', get the code and "
                       "enter it. A new code is needed for every video.")
    vinfo, t0 = access_log.video_info(video), time.time()
    opts = dict(src=src_choice, tgt=tgt_name, voice=voice, dub=do_dub, start=clip_start,
                len=clip_len, res=res)
    try:
        result = _translate_impl(video, src_choice, tgt_name, voice, do_dub, keep_bg, burn,
                                 clip_start, clip_len, res, progress)
    except Exception as e:
        access_log.write_record(person, vinfo, opts, "Failed", time.time() - t0, str(e)[:200])
        raise
    out, srt, dub_wav, status = result
    note = re.sub(r"[*`]", "", str(status))[:200]
    saved = access_log.write_record(person, vinfo, opts, "Done", time.time() - t0, note)
    where = ("records/translation_log.xlsx" if saved == access_log.XLSX else
             "records/translation_log_pending.csv (close the Excel file so it can be merged)")
    return out, srt, dub_wav, f"{status}\n\nRecorded for **{person['name']}** in `{where}`."


def _translate_impl(video, src_choice, tgt_name, voice, do_dub, keep_bg, burn,
                    clip_start=0, clip_len=0, res="Original resolution", progress=None):
    if not video:
        raise gr.Error("Upload a video first.")
    if not has_audio(video):
        raise gr.Error("This video has no audio track to translate.")
    purge_old_runs()
    tmp = tempfile.mkdtemp(prefix="ivt_")
    orig = f"{tmp}/orig.wav"

    progress(0.05, desc="Extracting audio")
    clip_start, clip_len = max(float(clip_start or 0), 0), max(float(clip_len or 0), 0)
    # Optional clip: only process [clip_start, clip_start + clip_len]; 0 length = whole video
    trim = []
    if clip_start > 0:
        trim += ["-ss", f"{clip_start:.3f}"]
    if clip_len > 0:
        trim += ["-t", f"{clip_len:.3f}"]
    run(["ffmpeg", "-y", *trim, "-i", video, "-vn", "-ac", "2", "-ar", "44100", orig])

    progress(0.15, desc="Transcribing speech")
    segs, detected = transcribe(orig, src_choice)
    if not segs:
        raise gr.Error("No speech found in this video.")
    src = WHISPER_TO_FLORES.get(detected)
    if src is None:
        raise gr.Error(f"Detected language '{detected}' isn't supported as a source. "
                       "Choose the spoken language manually.")

    tgt = LANGS[tgt_name]
    progress(0.4, desc=f"Translating to {tgt_name}")
    texts = translate([s["text"] for s in segs], src, tgt)

    srt_name = f"subtitles_{tgt_name}.srt"
    srt = os.path.join(tmp, srt_name)
    write_srt(segs, texts, srt)

    dub_wav, note = None, ""
    if do_dub:
        if tgt_name in TTS_LANGS:
            try:
                total_ms = len(AudioSegment.from_wav(orig))
                track = build_dub_track(segs, texts, voice, total_ms, tmp, progress, tgt_name)
                if keep_bg:
                    track = (AudioSegment.from_wav(orig) - 18).overlay(track)
                dub_wav = f"{tmp}/dub.wav"
                track.export(dub_wav, format="wav")
            except Exception as e:  # TTS failed: still deliver subtitles
                dub_wav = None
                note = f" Voice dubbing failed ({type(e).__name__}: {e}), so this video has subtitles only."
        else:
            note = f" Voice dubbing isn't available for {tgt_name}, so this video has subtitles only."

    progress(0.88, desc="Building final video")
    out = os.path.join(tmp, f"translated_{tgt_name}.mp4")
    video = os.path.abspath(video)
    cmd = ["ffmpeg", "-y", *trim, "-i", video]
    if dub_wav:
        cmd += ["-i", dub_wav]
    if not burn:
        cmd += ["-i", srt]
    cmd += ["-map", "0:v:0", "-map", "1:a:0" if dub_wav else "0:a?"]

    # Video filters: optional upscale to 4K (never downscale), then burned-in subtitles
    vf = []
    target = 3840 if res.startswith("4K") else 1920 if res.startswith("HD") else 0
    if target:
        w, h = video_size(video)
        k = target / max(w, h)
        # 4K only ever upscales; HD sets the long side to 1920 (up or down)
        if (k > 1 or target == 1920) and abs(k - 1) > 0.01:
            vf.append(f"scale={round(w * k / 2) * 2}:{round(h * k / 2) * 2}:flags=lanczos")
    if burn:
        # Relative name + cwd=tmp avoids ffmpeg's problems with Windows paths like C:\...
        vf.append(f"subtitles={srt_name}")
    if vf or trim:
        # Re-encode at high quality (CRF 16, near-lossless); a trimmed clip must be re-encoded
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]  # untouched video: original quality, very fast
    if not burn:
        cmd += ["-map", f"{2 if dub_wav else 1}:s:0", "-c:s", "mov_text"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", out]
    run(cmd, cwd=tmp)

    # Remove big intermediates; keep the final video and .srt for Gradio to serve
    for f in os.listdir(tmp):
        if f.endswith(".wav") and f != "dub.wav":
            os.remove(os.path.join(tmp, f))

    status = (f"Done. Spoken language: **{CODE_TO_NAME.get(src, src)}** → "
              f"**{tgt_name}**, {len(segs)} subtitle lines.{note}")
    return out, srt, dub_wav, status


# ---------- App ----------
SRC_CHOICES = ["Auto-detect"] + sorted(n for n, c in LANGS.items() if c in FLORES_TO_WHISPER)

RAIN_JS = r"""
() => {
  document.body.classList.add('dark');
  if (document.getElementById('word-rain')) return;
  const words = [
    "story","voice","dream","light","create","hello","language","magic",
    "కథ","భాష","స్వప్నం","వెలుగు","నమస్కారం","ప్రేమ","తెలుగు",
    "கதை","மொழி","கனவு","ஒளி","வணக்கம்","அன்பு","தமிழ்",
    "कहानी","भाषा","सपना","रोशनी","नमस्ते","प्यार","हिन्दी",
    "ಕಥೆ","ಭಾಷೆ","ಕನಸು","ಬೆಳಕು","ನಮಸ್ಕಾರ","ಪ್ರೀತಿ","ಕನ್ನಡ",
    "കഥ","ഭാഷ","സ്വപ്നം","വെളിച്ചം","നമസ്കാരം","മലയാളം",
    "গল্প","ভাষা","স্বপ্ন","আলো","বাংলা","વાર્તા","ભાષા","ਕਹਾਣੀ","ਭਾਸ਼ਾ","कथा","मराठी"
  ];
  const colors = ["255,122,198","167,139,250","94,200,255","255,170,120"];
  const cv = document.createElement('canvas');
  cv.id = 'word-rain';
  cv.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;';
  document.body.prepend(cv);
  const ctx = cv.getContext('2d');
  let W, H, drops = [];
  const font = '"Nirmala UI","Noto Sans Telugu","Noto Sans Tamil","Noto Sans Devanagari","Noto Sans Kannada","Segoe UI",sans-serif';
  function resize() {
    W = cv.width = window.innerWidth; H = cv.height = window.innerHeight;
    const n = Math.max(8, Math.floor(W / 65));
    drops = Array.from({length: n}, (_, i) => mk(i * (W / n) + Math.random() * 30, Math.random() * H));
    ctx.fillStyle = '#0b0b1a'; ctx.fillRect(0, 0, W, H);
  }
  function mk(x, y) {
    return {x, y, v: 0.15 + Math.random() * 0.35, size: 16 + Math.floor(Math.random() * 12),
            w: words[Math.floor(Math.random() * words.length)], c: colors[Math.floor(Math.random() * colors.length)],
            a: 0.40 + Math.random() * 0.45};
  }
  function frame() {
    ctx.fillStyle = 'rgba(11,11,26,0.35)'; ctx.fillRect(0, 0, W, H);
    for (const d of drops) {
      ctx.font = d.size + 'px ' + font;
      ctx.fillStyle = 'rgba(' + d.c + ',' + d.a + ')';
      ctx.fillText(d.w, d.x, d.y);
      d.y += d.v;
      if (d.y > H + 40) { Object.assign(d, mk(d.x, -20)); }
    }
    requestAnimationFrame(frame);
  }
  window.addEventListener('resize', resize);
  resize();
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) frame();
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.fuchsia, secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate, font=[gr.themes.GoogleFont("Poppins"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#0b0b1a", body_text_color="#e8e8f5",
    block_background_fill="rgba(255,255,255,0.045)", block_border_width="1px",
    block_border_color="rgba(255,255,255,0.10)", block_label_text_color="#c9b8ff",
    input_background_fill="rgba(255,255,255,0.07)", input_border_color="rgba(255,255,255,0.14)",
    button_primary_background_fill="linear-gradient(90deg,#ff3cac,#784ba0 55%,#2b86c5)",
    button_primary_background_fill_hover="linear-gradient(90deg,#ff5fbf,#8e5cb8 55%,#3c9ae0)",
    button_primary_text_color="white",
    button_secondary_background_fill="rgba(255,255,255,0.09)", button_secondary_text_color="#f0e8ff",
    body_background_fill_dark="#0b0b1a", body_text_color_dark="#e8e8f5",
    body_text_color_subdued="#b9b4d9", body_text_color_subdued_dark="#b9b4d9",
    block_background_fill_dark="rgba(255,255,255,0.045)", block_border_color_dark="rgba(255,255,255,0.10)",
    block_label_text_color_dark="#c9b8ff", block_title_text_color="#d9ccff",
    block_title_text_color_dark="#d9ccff", block_info_text_color="#b9b4d9",
    block_info_text_color_dark="#b9b4d9",
    input_background_fill_dark="rgba(255,255,255,0.07)", input_border_color_dark="rgba(255,255,255,0.14)",
    input_placeholder_color="#8d88b4", input_placeholder_color_dark="#8d88b4",
    checkbox_label_background_fill="rgba(255,255,255,0.07)", checkbox_label_background_fill_dark="rgba(255,255,255,0.07)",
    checkbox_label_text_color="#e8e8f5", checkbox_label_text_color_dark="#e8e8f5",
    checkbox_label_background_fill_selected="rgba(255,60,172,0.30)",
    checkbox_label_background_fill_selected_dark="rgba(255,60,172,0.30)",
    checkbox_label_text_color_selected="#ffffff", checkbox_label_text_color_selected_dark="#ffffff",
    button_secondary_background_fill_dark="rgba(255,255,255,0.09)", button_secondary_text_color_dark="#f0e8ff",
    button_primary_background_fill_dark="linear-gradient(90deg,#ff3cac,#784ba0 55%,#2b86c5)",
    button_primary_text_color_dark="white",
)

CSS = """
body {background: #0b0b1a !important;}
gradio-app {background: transparent !important; position: relative; z-index: 1; display: block;}
gradio-app .app, gradio-app .main, gradio-app .wrap.sidebar-parent, gradio-app > div {background: transparent !important;}
.gradio-container {background: transparent !important; position: relative; z-index: 1;
  max-width: 1100px !important; margin: auto;}
.gradio-container .block, .gradio-container .form {background: rgba(14,12,34,.80) !important;
  }
.hero {text-align:center; padding: 34px 10px 10px;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px;
  background: linear-gradient(90deg,#ff7ac6,#a78bfa,#5ec8ff); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.hero p {color:#b9b4d9; font-size: 1.05rem; margin: 8px 0 14px;}
.chips span {display:inline-block; margin: 3px 4px; padding: 4px 12px; border-radius: 999px;
  font-size: .82rem; color:#e9defc; background: rgba(167,139,250,.16); border:1px solid rgba(167,139,250,.35);}
.tab-wrapper, .tab-container {border-bottom-color: rgba(255,255,255,.1) !important;}
button[role=tab] {font-weight: 600; font-size: 1.02rem;}
button[role=tab][aria-selected=true] {color:#ff7ac6 !important; border-color:#ff7ac6 !important;}
button.primary, button.lg {box-shadow: 0 6px 22px rgba(255,60,172,.30); transition: transform .15s, box-shadow .15s;}
button.primary:hover {transform: translateY(-2px); box-shadow: 0 10px 28px rgba(255,60,172,.45);}
.step {margin-top: 18px; padding: 4px 16px 14px; border-radius: 18px;
  background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.09);}
.step h3 {margin: 12px 0 4px; background: linear-gradient(90deg,#ff7ac6,#a78bfa);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.gradio-container, .gradio-container * {--body-text-color:#e8e8f5;}
.gradio-container label, .gradio-container span, .gradio-container p, .gradio-container li,
.gradio-container td, .gradio-container th {color:#e8e8f5;}
.gradio-container input, .gradio-container textarea, .gradio-container select {color:#ffffff !important;
  background: rgba(255,255,255,.08) !important;}
.gradio-container span[data-testid="block-info"], .gradio-container .block-info {color:#c9b8ff !important;}
.hero h1 {color: transparent !important;}
.chips span {color:#e9defc !important;}
.step h3 {color: transparent !important;}
ul.options, .options, ul[role=listbox] {z-index: 99999 !important; max-height: 320px !important; overflow-y: auto !important;
  background: #1a1633 !important; opacity: 1 !important; border: 1px solid rgba(167,139,250,.45) !important;
  box-shadow: 0 12px 32px rgba(0,0,0,.6) !important;}
ul.options li, .options li, [role=option] {background: #1a1633 !important; color: #f2ecff !important; font-weight: 500;}
ul.options li:hover, ul.options li.active, [role=option]:hover, [role=option][aria-selected=true] {
  background: linear-gradient(90deg,#ff3cac66,#784ba066) !important; color: #ffffff !important;}
ul.options li span, [role=option] span {color: inherit !important;}
.gradio-container input[type=checkbox], .gradio-container input[type=radio] {
  -webkit-appearance: none !important; appearance: none !important; flex: none;
  width: 24px !important; height: 24px !important; border-radius: 7px !important;
  border: 2px solid #b79cff !important; background: #0b0b1a !important; background-image: none !important;
  cursor: pointer; position: relative; box-shadow: none !important;}
.gradio-container input[type=checkbox]:checked, .gradio-container input[type=radio]:checked {
  background-color: #ff3cac !important; border-color: #ff3cac !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5 12.5l4.5 4.5L19 7'/%3E%3C/svg%3E") !important;
  background-size: 18px 18px !important; background-position: center !important; background-repeat: no-repeat !important;}
.gradio-container label.selected, .gradio-container label:has(input:checked) {
  background: rgba(255,60,172,.22) !important; border-color: #ff3cac !important;}
.gradio-container label:has(input[type=checkbox]) {gap: 10px; font-weight: 500;}
[data-testid="block-label"], .gradio-container .block-label, label.block-label, span.block-label {
  background: linear-gradient(90deg,#3b1f72,#5b2a86) !important; color: #f2ecff !important;
  border: 1px solid rgba(167,139,250,.45) !important; box-shadow: none !important;}
[data-testid="block-label"] *, .gradio-container .block-label * {color: #f2ecff !important; background: transparent !important;}
[data-testid="block-label"] svg, .gradio-container .block-label svg {stroke: #f2ecff !important; color: #f2ecff !important;}
.gradio-container .wrap-inner, .gradio-container .secondary-wrap {
  background: linear-gradient(135deg,#1c1740,#241a52) !important; border: 1.5px solid rgba(167,139,250,.55) !important;
  border-radius: 14px !important; box-shadow: 0 4px 16px rgba(120,75,160,.25) !important; padding: 2px 6px !important;
  transition: border-color .15s, box-shadow .15s;}
.gradio-container .wrap-inner:hover, .gradio-container .wrap-inner:focus-within {
  border-color: #ff3cac !important; box-shadow: 0 0 0 3px rgba(255,60,172,.20), 0 4px 16px rgba(255,60,172,.25) !important;}
.gradio-container .wrap-inner input, .gradio-container .secondary-wrap input {
  background: transparent !important; border: none !important; outline: none !important; box-shadow: none !important;
  font-size: 1.02rem !important; font-weight: 600 !important; color: #ffffff !important; padding: 10px 8px !important;}
.gradio-container .secondary-wrap, .gradio-container .secondary-wrap:hover, .gradio-container .secondary-wrap:focus-within,
.gradio-container .wrap-inner .wrap-inner, .gradio-container .wrap-inner .secondary-wrap {
  background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;}
.gradio-container .container > .wrap {background: transparent !important; border: none !important; box-shadow: none !important;}
.gradio-container input[role=listbox] {border: none !important; background: transparent !important; box-shadow: none !important;}
.gradio-container .wrap-inner svg, .gradio-container .secondary-wrap svg {color: #ff7ac6 !important; fill: #ff7ac6 !important;}
.hero .by {margin-top: 12px; font-size: .95rem; color:#d9ccff;}
.hero .by b {background: linear-gradient(90deg,#ff7ac6,#a78bfa,#5ec8ff); -webkit-background-clip: text;
  background-clip: text; color: transparent !important;}
.corner-credit {position: fixed; right: 14px; bottom: 10px; z-index: 5; pointer-events: none;
  font-size: .8rem; color: rgba(233,222,252,.55); background: transparent; letter-spacing: .3px;
  text-shadow: 0 1px 6px rgba(0,0,0,.6);}
.corner-credit b {color: rgba(255,122,198,.75) !important; font-weight: 600;}
.foot {text-align:center; color:#7f7aa6; font-size:.85rem; padding: 18px 0 6px;}
"""

with gr.Blocks(title="Whisper Video Translator") as demo:
    gr.HTML('<div class="hero"><h1>Whisper Video Translator</h1>'
            '<p>Translate, dub and create videos in 22 Indian languages. Free, private, on your own machine.</p>'
            '<div class="chips"><span>Whisper</span><span>IndicTrans2</span><span>Indic Parler-TTS</span>'
            '<span>FFmpeg</span><span>4K output</span></div></div>')
    with gr.Tabs():
        with gr.Tab("Video translation"):
            gr.Markdown("Upload a video, choose a language, and get subtitles plus a dubbed version.")
            video_in = gr.Video(label="Upload video")
            with gr.Row():
                src_in = gr.Dropdown(SRC_CHOICES, value="Auto-detect", label="Language spoken in the video")
                tgt_in = gr.Dropdown(sorted(LANGS), value="Hindi", label="Translate into")
            voice_in = gr.State("Male")  # the dubbing voice is always Male
            dub_in = gr.Checkbox(value=True, label="Dub the audio")
            bg_in = gr.State(False)  # original sound is not kept in the background
            burn_in = gr.Checkbox(value=True, label="Show subtitles on the video (burned in)")
            with gr.Row():
                start_in = gr.Number(value=0, minimum=0, label="Start at (seconds)")
                len_in = gr.Number(value=0, minimum=0,
                                   label="Process only this many seconds (0 = full video, e.g. 10 or 13 to test)")
            res_in = gr.Radio(["Original resolution", "HD (1080p)", "4K HD (2160p)"], value="Original resolution",
                              label="Output quality (original = no quality loss; HD = 1920 px wide; 4K HD = 3840 px, upscaled, slower)")
            with gr.Column(elem_classes="step"):
                gr.Markdown("### Who is translating? (email code needed for every video)")
                with gr.Row():
                    who_name = gr.Textbox(label="Your name", placeholder="e.g. Abhishek")
                    who_email = gr.Textbox(label="Your email (Gmail)", placeholder="you@gmail.com")
                who_desc = gr.Textbox(label="Short description of the video", lines=2, max_lines=3,
                                      placeholder="e.g. Class lecture on photosynthesis, Telugu to English")
                gr.Examples(
                    examples=[["Class lecture on photosynthesis, Telugu to English, for school students"],
                              ["Product demo video for our new app, English to Hindi, 2 minutes"],
                              ["Family function wishes message, Hindi to Tamil, personal use"]],
                    inputs=[who_desc], label="Sample descriptions (click one to fill the box, then edit)")
                who_method = gr.Radio(["Code shown on this PC's screen", "Authenticator app (works offline)",
                                       "Email code (needs Gmail setup)"],
                                      value="Code shown on this PC's screen", label="How do you want to get your code?")
                with gr.Row():
                    who_send = gr.Button("Next: get my code")
                    who_code = gr.Textbox(label="6-digit code", max_length=6)
                    who_verify = gr.Button("Verify code")
                who_qr = gr.Image(label="Scan this once in your authenticator app", visible=False,
                                  show_label=True, interactive=False, height=260, buttons=[])
                who_status = gr.Markdown()
                who_token = gr.State(None)
            btn = gr.Button("Translate", variant="primary")
            video_out = gr.Video(label="Translated video")
            srt_out = gr.File(label="Subtitle file (.srt)")
            dub_out = gr.File(label="Dubbed audio (.wav) - use this for lip-sync, see LIPSYNC.md")
            status_out = gr.Markdown()
            def _send(token, name, email, desc, method):
                if method.startswith("Authenticator"):
                    token, ok, msg, qr = access_log.totp_start(token, name, email, desc)
                else:
                    token, ok, msg = access_log.send_code(
                        token, name, email, desc, "screen" if method.startswith("Code shown") else "email")
                    qr = None
                return (token, ("" if ok else "**Problem:** ") + msg,
                        gr.update(value=qr, visible=qr is not None))

            def _verify(token, code):
                ok, msg = access_log.verify(token, code)
                # hide the QR once verified (it contains the secret key)
                return ("" if ok else "**Problem:** ") + msg, (gr.update(value=None, visible=False) if ok else gr.update())

            who_send.click(_send, [who_token, who_name, who_email, who_desc, who_method],
                           [who_token, who_status, who_qr])
            who_verify.click(_verify, [who_token, who_code], [who_status, who_qr])
            btn.click(process, [video_in, src_in, tgt_in, voice_in, dub_in, bg_in, burn_in, start_in, len_in,
                                res_in, who_token],
                      [video_out, srt_out, dub_out, status_out], show_progress_on=video_out)
    gr.HTML('<div class="foot">Runs fully on your computer. Nothing is uploaded anywhere.</div>'
            '<div class="corner-credit">Developed and created by <b>Abhishek Surya Doppalapudi</b></div>')

if __name__ == "__main__":
    # Set SHARE=1 to get a temporary public link (e.g. when running on Google Colab)
    share = os.environ.get("SHARE", "0") == "1"
    print(f"Running on: {DEVICE.upper()}" + ("" if DEVICE == "cuda" else
          "  (no GPU found - dubbing will be slow; try short clips first)"))
    demo.queue().launch(share=share, inbrowser=not share, theme=THEME, css=CSS,
                          head="<script>(function(){ var go=function(){ try { ("
                               + RAIN_JS + ")(); document.documentElement.setAttribute('data-rain','ok'); }"
                               " catch(e) { document.documentElement.setAttribute('data-rain','err:'+e); } };"
                               " if (document.readyState==='loading') document.addEventListener('DOMContentLoaded', go);"
                               " else go(); })();</script>")
