# EmotionSense

**EmotionSense** is a real-time facial emotion detection system built with **TensorFlow (Keras)** and **OpenCV**.
It uses a Convolutional Neural Network (CNN) trained on the **FER-2013** dataset to recognize emotions from live webcam video feeds.

> Detect emotions in real time, save reactions, and stream to video-call applications via a virtual webcam.

---

## Key Highlights

* Real-time emotion detection using TensorFlow and OpenCV.
* Flexible model loader supporting both full saved models and weights-only HDF5 files.
* Interactive display mode with live emotion labels and top-left statistics overlay.
* Smooth control system with keyboard shortcuts for overlays and stop commands.
* Optional virtual webcam support using `pyvirtualcam` for integration with Zoom, Teams, or OBS.
* Verified on Windows 10 using a Python 3.10 virtual environment for TensorFlow–NumPy compatibility.

---

## Tech Stack

| Category    | Tools / Libraries          |
| ----------- | -------------------------- |
| Language    | Python 3.10                |
| Frameworks  | TensorFlow (Keras), OpenCV |
| Utilities   | NumPy, pyvirtualcam        |
| Dataset     | FER-2013                   |
| Environment | Windows 10 (tested)        |

---

## Quick Start (Windows / PowerShell)

### 1. Set up the environment

```powershell
cd C:\Users\hsdeeksha\OneDrive\Desktop\EmotionSense\Emotion-detection
python -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run real-time detection

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0
```

**Controls:**

* Press `o` to toggle overlays (labels and stats)
* Press `q`, `Q`, or `ESC` to stop
* Press `Ctrl + C` in the terminal to stop

---

## Save-on-Detection Mode

Automatically save frames or video clips when specific emotions are detected.

**Example:**

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0 ^
--save-on Happy,Sad --save-dir .\saved --clip-length 3.0 --pre-clip 1.0 --cooldown 5 --vcam-fps 10
```

**Flags:**

| Flag                 | Description                                     |
| -------------------- | ----------------------------------------------- |
| `--save-on`          | Comma-separated emotion labels to trigger saves |
| `--save-dir`         | Directory to save output files                  |
| `--clip-length`      | Length of each saved clip (seconds)             |
| `--pre-clip`         | Seconds of footage before the trigger           |
| `--cooldown`         | Time between saves for the same emotion         |
| `--save-images-only` | Save still images only (no clips)               |

---

## Virtual Webcam Mode

Use the `--virtual` flag to stream the processed video feed into apps like Zoom, Teams, or OBS.

**Example:**

```powershell
python .\src\emotions.py --mode display --model .\src\model.h5 --camera 0 --virtual
```

Requires `pyvirtualcam` (installed automatically via `requirements.txt`).

---

## Training and Model

* CNN model trained on the **FER-2013** dataset.
* You can retrain or fine-tune using the following command:

	```powershell
	python .\src\emotions.py --mode train
	```
* Example accuracy plot:
	![Training Accuracy](imgs/accuracy.png)

---

## Repository Structure

```
EmotionSense/
│
├── src/
│   ├── emotions.py                  # Main script (train / display modes)
│   ├── model.h5                     # Trained model (weights or full)
│   ├── haarcascade_frontalface_default.xml
│   ├── dataset_prepare.py           # Dataset preparation utility
│
├── imgs/
│   └── accuracy.png                 # Example training plot
│
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue                   | Solution                                                      |
| ----------------------- | ------------------------------------------------------------- |
| TensorFlow import error | Use Python 3.8–3.10 and `numpy==1.23.5`, `tensorflow==2.9.3`  |
| Haar cascade not found  | Ensure `haarcascade_frontalface_default.xml` exists in `src/` |
| Webcam not detected     | Adjust `--camera` index (try 1, 2, etc.)                      |
| Laggy preview           | Reduce frame resolution or disable overlays using the `o` key |

---

## Next Steps

* Convert the model to **TensorFlow.js** for browser-based inference.
* Integrate a **Streamlit dashboard** for emotion analytics.
* Make the save-on-detection process **non-blocking** for smoother performance.
* Add **cloud sync** support for emotion data summaries.

---

## Author

**Deeksha H S**
GitHub: [hsdeeksha](https://github.com/hsdeeksha)

---

## (Optional) Demo

You can add a short GIF or screenshot here once available:

```markdown
![EmotionSense Live Demo](imgs/demo.gif)
```

  