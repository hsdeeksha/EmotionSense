# EmotionSense

EmotionSense is a lightweight, real-time facial emotion detection tool using a small CNN (Keras) and OpenCV.
It supports both full saved models and weights-only HDF5 files, an interactive display mode with compact overlays,
and an opt-in save-on-detection feature for saving images or short clips when configured emotions appear.

This README reflects the current code in `src/` (training utilities, live display, and dataset preparation helper).

## Highlights

- Real-time emotion detection from a webcam using TensorFlow (Keras) and OpenCV
- Flexible model loading: attempts `load_model()` then falls back to `model.load_weights()` for HDF5 weight files
- Interactive display with keyboard controls and a compact live-stats overlay
- Save-on-detection: save images and short clips (prebuffer + post frames) when chosen emotions are detected
- Optional virtual webcam output (via `pyvirtualcam`) to feed processed video into Zoom/Teams/OBS

## Quick facts

- Recommended Python: 3.10 (Windows)
- Typical dependencies: `tensorflow`, `opencv-python`, `numpy`, `matplotlib`, and optional `pyvirtualcam`
- Project layout (important files):

```
EmotionSense/
├─ src/
│  ├─ emotions.py                  # Main script: --mode train|display
│  ├─ dataset_prepare.py           # Convert FER-2013 CSV to train/test folders
│  ├─ haarcascade_frontalface_default.xml
│  └─ model.h5                     # Trained weights or full model (optional)
├─ imgs/
│  └─ accuracy.png
├─ requirements.txt
└─ README.md
```

## Install (Windows / PowerShell)

Open PowerShell and run:

```powershell
cd C:\Users\hsdeeksha\OneDrive\Desktop\EmotionSense\Emotion-detection
python -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- If you encounter TensorFlow import errors on newer Python/NumPy versions, use Python 3.8–3.10 and `numpy==1.23.5` with `tensorflow==2.9.3` (these are known-compatible versions).

## Run the live detector

Basic display mode (webcam index 0):

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0
```

Keyboard controls while running:
- `o` — toggle overlays (labels + live stats)
- `q`, `Q`, `Esc` — quit the app
- `Ctrl+C` in the terminal — also quits

## Save-on-Detection (images and clips)

The display mode supports automatically saving still images and short clips when selected emotion labels appear.

Example (PowerShell):

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0 \
  --save-on Happy,Sad --save-dir .\saved --clip-length 3.0 --pre-clip 1.0 --cooldown 5 --vcam-fps 10
```

Important flags:

- `--save-on` — comma-separated emotion labels to trigger saves (e.g. "Happy,Sad"). Leave empty to disable.
- `--save-dir` — directory where images and clips are written
- `--clip-length` — total seconds per saved clip (includes `--pre-clip`)
- `--pre-clip` — seconds of footage to include from before the trigger (must be <= clip-length)
- `--cooldown` — minimum seconds between saves for the same emotion
- `--min-count` — minimum recent occurrences of the emotion in the in-memory history required to trigger
- `--save-images-only` — save only still images (disable clip writing)

Behavior notes:
- The implementation keeps a small prebuffer (frames before the trigger) and then captures subsequent frames to write a clip. By default the clip writer runs synchronously and may block the main loop while writing; this is intentional (simple and robust) but can be converted to a background writer queue for non-blocking operation.

## Virtual webcam output

Use `--virtual` to attempt to open a virtual camera (requires `pyvirtualcam`). Example:

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0 --virtual --vcam-fps 15
```

If `pyvirtualcam` is not installed or fails to open, the script falls back to normal preview-only mode.

## Training / dataset

- To create dataset folders from FER-2013 CSV, use `src/dataset_prepare.py` (it reads `fer2013.csv` and writes `data/train` and `data/test`).
- To train the model from `src/emotions.py` run:

```powershell
python .\src\emotions.py --mode train
```

Notes: training code expects `data/train` and `data/test` folders with 48x48 grayscale images arranged by class.

## Troubleshooting

- Haar cascade missing: ensure `src/haarcascade_frontalface_default.xml` exists. The script loads it from the `src` directory.
- Model loading: the script will first try `tensorflow.keras.models.load_model(path)` (full model). If that fails it will load weights into the built architecture — this supports weights-only `model.h5` files.
- Webcam errors / wrong index: try different `--camera` indices (0, 1, 2...).
- TensorFlow / NumPy version issues: use Python 3.8–3.10 and install `numpy==1.23.5` with a matching TensorFlow (e.g. `tensorflow==2.9.3`).

## Best practices (repo hygiene)

- Do NOT commit virtual environments or site-packages to the repository. If you accidentally committed a virtualenv (for example `.venv310/`), remove it from the repo and add it to `.gitignore`. Pushing those files bloats the repository.

Add to `.gitignore` (example):

```
# Virtual environments
.venv/
venv/
.venv310/
src/.venv310/

# Python cache
__pycache__/
*.pyc
```

## Next steps (suggested)

- Make the save-on-detection writer non-blocking by adding a background worker / queue to perform disk writes.
- Add a small web dashboard (Streamlit or Flask) to visualize emotion summaries and saved clips.
- Convert model to TensorFlow.js for browser inference.

## License & Author

Author: Deeksha H S
Repository: https://github.com/hsdeeksha/EmotionSense
