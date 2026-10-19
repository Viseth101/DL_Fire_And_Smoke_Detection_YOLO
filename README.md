# Smoke and Fire Detection with YOLO

A deep-learning project for detecting **smoke** and **fire** in images and video. It includes a Jupyter notebook for dataset exploration, model training, and evaluation, plus a standalone script for running detections on video files, webcams, and IP camera streams.

## Project contents

| Path | Description |
| --- | --- |
| `Project/DL_Term_Project_Notebook.ipynb` | Dataset analysis, preprocessing, YOLO model comparisons, tuning, test-set evaluation, and deployment notes. |
| `Project/deploy_detector.py` | Interactive and command-line inference application with an optional live preview, annotated video output, and hazard alerts. |
| `Project/Video Deploy/Models/yolo26s_best.onnx` | Portable ONNX model used by the deployment script by default. |
| `Project/Video Deploy/Models/yolo26s_best.pt` | PyTorch model weights. |
| `Project/Video Deploy/Source/` | Sample videos for testing and other video inputs. |
| `Project/runs/detect/Smoke_Fire_Detection/` | Training and evaluation outputs, including model weights, plots, and metrics. |

The detector uses class `0` for smoke and class `1` for fire. The dataset itself is not included; see [Training and evaluation](#training-and-evaluation) to obtain it.

## Requirements

- Python with `pip`
- A working video source (a sample video, webcam, or accessible IP camera stream)
- Internet access the first time dependencies are installed

The deployment script checks for and installs its missing Python packages at startup. To install them yourself in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python onnx onnxruntime Pillow requests
```

For the notebook's Kaggle dataset download, also install:

```powershell
python -m pip install kagglehub
```

## Run detection

From the repository root, launch the interactive source selector:

```powershell
python .\Project\deploy_detector.py
```

Or pass an input directly:

```powershell
# Process a sample video and save the annotated result
python .\Project\deploy_detector.py --source ".\Project\Video Deploy\Source\sample_fire3.mp4" --save

# Use the default webcam
python .\Project\deploy_detector.py --source 0

# Connect to an IP camera stream
python .\Project\deploy_detector.py --source "rtsp://camera-address/stream"

# Run without a preview window (saves output video)
python .\Project\deploy_detector.py --source ".\Project\Video Deploy\Source\sample_fire3.mp4" --no-gui
```

The deployment script automatically looks for `Project/Video Deploy/Models/yolo26s_best.onnx`. To use different weights, pass `--model` with a `.onnx` or `.pt` file:

```powershell
python .\Project\deploy_detector.py --model ".\path\to\weights.pt" --source ".\Project\Video Deploy\Source\sample_fire3.mp4"
```

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--model PATH` | Automatically selected | Path to custom ONNX or PyTorch weights. |
| `--source SOURCE` | Interactive selection | Video path, webcam index such as `0`, or RTSP/HTTP stream URL. |
| `--conf VALUE` | `0.10` | Minimum detection confidence, from `0` to `1`. |
| `--buffer FRAMES` | `5` | Number of recent frames used to confirm a hazard. |
| `--hits COUNT` | `4` | Detections in the buffer required to confirm a hazard. |
| `--cooldown SECONDS` | `60` | Minimum interval between alerts. |
| `--save` | Off | Save an annotated output video. Headless (`--no-gui`) mode saves output automatically. |
| `--enable-alerts` | Off | Send confirmed hazard alerts to Telegram. |
| `--no-gui` | Off | Run without opening a preview window. |

Annotated videos are written to `Project/Video Deploy/Output/`. When an alert is triggered, a snapshot is saved to `Project/Video Deploy/Alerts_Log/`. The detector uses a rolling window to avoid triggering on a single-frame detection.

### Telegram alerts

Alerts are printed and snapshots are saved locally by default. To also send Telegram photo alerts, set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the environment and pass `--enable-alerts`. For example, in PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
python .\Project\deploy_detector.py --source ".\Project\Video Deploy\Source\sample_fire3.mp4" --enable-alerts
```

Keep bot tokens private; do not commit credentials to the repository.

## Training and evaluation

Open `Project/DL_Term_Project_Notebook.ipynb` in Jupyter or VS Code and run its cells in order. The notebook covers exploratory data analysis, preprocessing and augmentation, YOLO baseline comparisons, hyperparameter tuning, and evaluation on a held-out test set.

The notebook downloads the smoke-and-fire dataset with KaggleHub (`sayedgamal99/smoke-fire-detection-yolo`). Install `kagglehub` first, and update any dataset or output paths in the notebook to match your environment before running training and evaluation cells. Training requires the dataset and can require substantial time and compute; a GPU is recommended.

## Notes

- The deployment application is intended for demonstration and experimentation. Do not rely on its predictions as the sole means of fire or smoke detection or emergency response.
- Model accuracy and detection speed depend on the model weights, input, hardware, and confidence settings.
