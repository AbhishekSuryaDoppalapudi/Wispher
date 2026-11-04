import os, gc, subprocess, tempfile
import torch, numpy as np, soundfile as sf
import whisper
import gradio as gr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    from IndicTransToolkit import IndicProcessor
from parler_tts import ParlerTTSForConditionalGeneration

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

# Languages Indic Parler-TTS can speak. Others still get translated subtitles.
TTS_LANGS = {"Assamese", "Bengali", "Bodo", "Dogri", "English", "Gujarati", "Hindi", "Kannada",
             "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", "Nepali",
             "Odia", "Punjabi", "Sanskrit", "Santali", "Tamil", "Telugu", "Urdu"}

# Spoken languages Whisper can recognise in the uploaded video
WHISPER_TO_FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ur": "urd_Arab", "as": "asm_Beng", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
}
FLORES_TO_WHISPER = {v: k for k, v in WHISPER_TO_FLORES.items()}

VOICES = {
    "Female": "A female speaker delivers clear, natural speech at a moderate pace and pitch. "
              "The recording is of very high quality, with the voice sounding close up and clear.",
    "Male": "A male speaker delivers clear, natural speech at a moderate pace and pitch. "
            "The recording is of very high quality, with the voice sounding close up and clear.",
}


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])


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
        m = ParlerTTSForConditionalGeneration.from_pretrained(name).to(DEVICE)
        _tts["model"] = m
        _tts["tok"] = AutoTokenizer.from_pretrained(name)
        _tts["desc_tok"] = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
    return _tts


def speak(text, voice, out_path):
    t = load_tts()
    d = t["desc_tok"](VOICES[voice], return_tensors="pt").to(DEVICE)
    p = t["tok"](text, return_tensors="pt").to(DEVICE)
    torch.manual_seed(42)  # keeps the voice consistent across lines
    with torch.no_grad():
        g = t["model"].generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                prompt_input_ids=p.input_ids,
                                prompt_attention_mask=p.attention_mask)
    audio = g.cpu().numpy().squeeze().astype(np.float32)
    sf.write(out_path, audio, t["model"].config.sampling_rate)


def fit_to_slot(wav_path, slot_sec, out_path, max_speed=1.6):
    """Speed up a line slightly if it's longer than the gap available."""
    dur = sf.info(wav_path).duration
    speed = 1.0
    if slot_sec > 0.3 and dur > slot_sec:
        speed = min(dur / slot_sec, max_speed)
    run(["ffmpeg", "-y", "-i", wav_path, "-filter:a", f"atempo={speed:.3f}",
         "-ar", "44100", "-ac", "1", out_path])


def build_dub_track(segs, texts, voice, total_ms, tmp, progress):
    track = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    n = len(segs)
    for i, (s, txt) in enumerate(zip(segs, texts)):
        progress(0.55 + 0.3 * i / n, desc=f"Generating voice {i + 1}/{n}")
        if not txt.strip():
            continue
        raw, fit = f"{tmp}/seg{i}.wav", f"{tmp}/fit{i}.wav"
        speak(txt, voice, raw)
        nxt = segs[i + 1]["start"] if i + 1 < n else s["end"] + 1.0
        fit_to_slot(raw, nxt - s["start"], fit)
        track = track.overlay(AudioSegment.from_wav(fit), position=int(s["start"] * 1000))
    return track


# ---------- Full pipeline ----------
def process(video, src_choice, tgt_name, voice, do_dub, keep_bg, burn, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video first.")
    tmp = tempfile.mkdtemp()
    orig = f"{tmp}/orig.wav"

    progress(0.05, desc="Extracting audio")
    run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100", orig])

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
            total_ms = len(AudioSegment.from_wav(orig))
            track = build_dub_track(segs, texts, voice, total_ms, tmp, progress)
            if keep_bg:
                track = (AudioSegment.from_wav(orig) - 18).overlay(track)
            dub_wav = f"{tmp}/dub.wav"
            track.export(dub_wav, format="wav")
        else:
            note = f" Voice dubbing isn't available for {tgt_name}, so this video has subtitles only."

    progress(0.88, desc="Building final video")
    out = os.path.join(tmp, f"translated_{tgt_name}.mp4")
    video = os.path.abspath(video)
    cmd = ["ffmpeg", "-y", "-i", video]
    if dub_wav:
        cmd += ["-i", dub_wav]
    if not burn:
        cmd += ["-i", srt]
    cmd += ["-map", "0:v:0", "-map", "1:a:0" if dub_wav else "0:a?"]
    if burn:
        # Relative name + cwd=tmp avoids ffmpeg's problems with Windows paths like C:\...
        cmd += ["-vf", f"subtitles={srt_name}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23"]
    else:
        sub_idx = 2 if dub_wav else 1
        cmd += ["-map", f"{sub_idx}:s:0", "-c:s", "mov_text", "-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", out]
    run(cmd, cwd=tmp)

    status = (f"Done. Spoken language: **{CODE_TO_NAME.get(src, src)}** → "
              f"**{tgt_name}**, {len(segs)} subtitle lines.{note}")
    return out, srt, status


# ---------- App ----------
SRC_CHOICES = ["Auto-detect"] + sorted(n for n, c in LANGS.items() if c in FLORES_TO_WHISPER)

with gr.Blocks(title="Indic Video Translator") as demo:
    gr.Markdown("## Indic Video Translator\nUpload a video, choose a language, "
                "and get subtitles plus a dubbed version.")
    video_in = gr.Video(label="Upload video")
    with gr.Row():
        src_in = gr.Dropdown(SRC_CHOICES, value="Auto-detect", label="Language spoken in the video")
        tgt_in = gr.Dropdown(sorted(LANGS), value="Hindi", label="Translate into")
    voice_in = gr.Radio(list(VOICES), value="Female", label="Dubbing voice")
    dub_in = gr.Checkbox(value=True, label="Dub the audio")
    bg_in = gr.Checkbox(value=True, label="Keep the original sound quietly in the background")
    burn_in = gr.Checkbox(value=True, label="Show subtitles on the video (burned in)")
    btn = gr.Button("Translate", variant="primary")
    video_out = gr.Video(label="Translated video")
    srt_out = gr.File(label="Subtitle file (.srt)")
    status_out = gr.Markdown()
    btn.click(process, [video_in, src_in, tgt_in, voice_in, dub_in, bg_in, burn_in],
              [video_out, srt_out, status_out])

if __name__ == "__main__":
    # Set SHARE=1 to get a temporary public link (e.g. when running on Google Colab)
    share = os.environ.get("SHARE", "0") == "1"
    print(f"Running on: {DEVICE.upper()}" + ("" if DEVICE == "cuda" else
          "  (no GPU found - dubbing will be slow; try short clips first)"))
    demo.queue().launch(share=share, inbrowser=not share, debug=True)
