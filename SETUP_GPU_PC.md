# Setting up on a dedicated PC (RTX 3060 12 GB)

Developed and created by Abhishek Surya Doppalapudi.

## 1. One-time install (needs internet)
1. Windows 10/11 with the latest NVIDIA driver (nvidia.com/drivers).
2. Install **Python 3.12** (python.org, tick "Add to PATH") and **Git**.
3. Install FFmpeg: `winget install ffmpeg`, then open a new terminal. Check `ffmpeg -version`.
4. Get the code:
   ```
   git clone https://github.com/AbhishekSuryaDoppalapudi/Wispher.git
   cd Wispher
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install the GPU build of PyTorch first (CUDA 12.1):
   ```
   python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r requirements.txt
   ```
   (requirements.txt would otherwise install the CPU build.)
6. Check the GPU is seen: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
7. If Windows says "An Application Control policy has blocked this file", turn off
   Smart App Control (Windows Security > App & browser control), then try again.

## 2. Download the models (needs internet, once)
- Make a free Hugging Face account, accept the licence on `ai4bharat/indic-parler-tts`
  and the IndicTrans2 model pages, then run `huggingface-cli login`.
- Start the app once (`python app.py`) and translate a short clip with dubbing.
  This downloads about 6 GB into `C:\Users\<you>\.cache\huggingface` and `...\.cache\whisper`.

## 3. Run without internet afterwards
- Set `HF_HUB_OFFLINE=1` (System Properties > Environment Variables) so it never tries to connect.
- Verification: use "Code shown on this PC's screen" or "Authenticator app". Email code needs internet.
- Start: `.venv\Scripts\python.exe app.py`, open http://127.0.0.1:7860.
- To reach it from other PCs on the same network, ask for the network option first; only do this
  on a network you trust.

## 4. Copying to an offline PC instead
Do steps 1-2 on any PC with internet, then copy the project folder (including `.venv`)
and the two `.cache` folders (`huggingface`, `whisper`) to the same paths on the dedicated PC.

## 5. Back up
Copy the `records` folder regularly. It holds the Excel log of who translated what.
The `.env` file (only for Gmail) is private and is not in Git.
