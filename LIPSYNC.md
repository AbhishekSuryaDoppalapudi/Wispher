# Lip-sync (optional, needs a GPU)

The app matches the *timing* of the dub to the speaker. To also redraw the speaker's mouth so it
matches the new voice, run a lip-sync model on a GPU. Free option: Google Colab (T4).

Model: [LatentSync](https://github.com/bytedance/LatentSync) 1.5 (needs about 8 GB GPU memory;
1.6 needs 18 GB, more than a free T4). Check its license before any commercial use.

> These steps follow the LatentSync README and have **not** been tested by this project yet.

## Steps

1. In this app, translate your video with **"Dub the audio" ticked** and
   **"Show subtitles on the video" unticked** (burned-in subtitles would be redrawn on the face).
2. Download the **Dubbed audio (.wav)** the app produces. Keep your **original video** too.
3. Open https://colab.research.google.com, choose **Runtime > Change runtime type > T4 GPU**.
4. Run in a cell:
   ```
   !git clone https://github.com/bytedance/LatentSync.git
   %cd LatentSync
   !source setup_env.sh
   !python gradio_app.py
   ```
   Open the public link it prints.
5. Upload the original video and the dubbed `.wav`, and run. Higher `inference_steps` (20-50) gives
   better quality, higher `guidance_scale` (1.0-3.0) gives tighter sync but can cause jitter.

If the Colab steps fail, the project's Hugging Face Space
(https://huggingface.co/spaces/fffiloni/LatentSync) offers the same model in a browser.

## Limits

- It works best on a clear, front-facing face. Side views or several faces can fail.
- It changes only the mouth area; the audio length must match the video length (the app builds the
  dub to the original length).
