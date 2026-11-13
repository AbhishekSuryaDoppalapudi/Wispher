"""Run this on a GPU machine (cloud / Colab Pro) to turn scenes.json into real video clips.

It calls LTX-2's `ltx_pipelines.distilled` once per scene (command taken from the LTX-2 README)
and saves clips/NNN.mp4. It skips clips that already exist, so you can stop and resume.
NOT TESTED here: this laptop has no NVIDIA GPU. Uses only the standard library.

Setup on the GPU machine (see third_party/LTX-2/README.md, ~66 GiB of weights):
    cd third_party/LTX-2 && uv sync --extra natten
    hf auth login
    hf download Lightricks/LTX-2.5 <the five files from the README> --local-dir models/ltx-2.5

Then, from the project root:
    python story_video/gpu_clips.py --project mystory
Copy story_work/mystory/clips/ back to your laptop and run `assemble`.
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LTX = os.path.join(ROOT, "third_party", "LTX-2")

p = argparse.ArgumentParser()
p.add_argument("--project", required=True)
p.add_argument("--models", default=os.path.join(LTX, "models", "ltx-2.5"))
p.add_argument("--frames", type=int, default=121, help="121 frames at 24 fps is about 5 s")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--low-vram", action="store_true", help="add --quantization fp8-cast --offload cpu")
a = p.parse_args()

work = os.path.join(ROOT, "story_work", a.project)
scenes = json.load(open(os.path.join(work, "scenes.json"), encoding="utf-8"))
os.makedirs(os.path.join(work, "clips"), exist_ok=True)
m = a.models
for s in scenes:
    out = os.path.join(work, "clips", f"{s['id']:03d}.mp4")
    if os.path.exists(out):
        continue
    cmd = ["uv", "run", "python", "-m", "ltx_pipelines.distilled",
           "--transformer-path", f"{m}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
           "--text-encoder-path", f"{m}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
           "--video-vae-path", f"{m}/vae/ltx-2.5-video-vae-bf16.safetensors",
           "--audio-vae-path", f"{m}/vae/ltx-2.5-audio-vae-bf16.safetensors",
           "--spatial-upsampler-path",
           f"{m}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
           "--num-frames", str(a.frames), "--seed", str(a.seed + s["id"]),
           "--output-path", out, "--prompt", s["visual"]]
    if a.low_vram:
        cmd += ["--quantization", "fp8-cast", "--offload", "cpu"]
    print(f"Scene {s['id']}/{len(scenes)}: {s['visual'][:80]}...")
    if subprocess.run(cmd, cwd=LTX).returncode != 0:
        sys.exit(f"LTX-2 failed on scene {s['id']}. Fix and re-run; finished clips are kept.")
print("All clips done. Copy story_work/%s/clips to your laptop and run `assemble`." % a.project)
