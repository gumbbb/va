# Build with: DOCKER_BUILDKIT=1 docker compose build (or docker build) so pip/apt cache mounts work
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    curl \
    libopenblas0 \
    portaudio19-dev \
    alsa-utils \
    libasound2-dev \
    libasound2-plugins \
    libpulse0 \
    pulseaudio-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "setuptools<70" && \
    python3 -m pip install numpy==1.26.1 && \
    python3 -m pip install psutil==7.0.0 scipy==1.15.3 soundfile==0.13.1 && \
    python3 -m pip install --no-build-isolation openai-whisper==20240927 && \
    python3 -m pip install onnx-graphsurgeon==0.5.7 && \
    python3 -m pip install openwakeword==0.5.1 && \
    python3 -m pip install silero-vad==5.1.2 && \
    python3 -m pip install webrtcvad && \
    python3 -m pip install sounddevice==0.5.1 PyAudio pydub python-dotenv && \
    python3 -m pip install colorlog "espeakng-loader>=0.2.4" "phonemizer-fork>=3.3.2" && \
    python3 -m pip install requests websocket-client

COPY torch2trt /app/torch2trt_src
COPY whisper_trt /app/whisper_trt_src
COPY wheels /app/wheels
COPY . /app

RUN test -f /app/wheels/torch-2.8.0-cp310-cp310-linux_aarch64.whl || \
    (echo "ERROR: torch wheel required at /app/wheels/torch-2.8.0-cp310-cp310-linux_aarch64.whl" && exit 1)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    /app/wheels/torch-2.8.0-cp310-cp310-linux_aarch64.whl \
    /app/wheels/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl \
    /app/wheels/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl

RUN test -f /app/wheels/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl || \
    (echo "ERROR: onnxruntime_gpu wheel required at /app/wheels/" && exit 1)
RUN cp /app/wheels/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl /tmp/ && \
    python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install /tmp/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
RUN python3 -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'ONNX Runtime GPU required'"

RUN mkdir -p /app/agent_assets/models && \
    python3 -c "import silero_vad, os, shutil; src=os.path.join(os.path.dirname(silero_vad.__file__),'data','silero_vad.onnx'); dest='/app/agent_assets/models/silero_vad.onnx'; shutil.copy(src, dest); print('Copied silero_vad.onnx to /app/agent_assets/models')"

ENV XDG_CACHE_HOME=/app/cache

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -d /app/kokoro_tts/kokoro-onnx ]; then python3 -m pip install -e /app/kokoro_tts/kokoro-onnx; fi

ENV PYTHONPATH=/app
WORKDIR /app

CMD ["/bin/bash", "-c", "cd /app/torch2trt && python3 setup.py install && cd /app/whisper_trt && python3 setup.py install && exec python3 /app/main.py"]
