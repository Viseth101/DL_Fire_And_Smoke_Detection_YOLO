FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr so logs appear in real-time
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

# Install minimal Linux system libraries required by OpenCV & ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer for fast rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model weights
COPY src/ ./src/
COPY models/ ./models/
COPY deploy_detector.py .

# Default entrypoint runs the detector in headless server mode with alerts enabled
ENTRYPOINT ["python", "deploy_detector.py", "--no-gui", "--enable-alerts"]