# Deploy & Transfer Guide

Tài liệu tập hợp: nén project, copy sang máy host, push/pull Docker images (bao gồm voice_processing).

---

## 1. Nén project (compress)

### Script có sẵn: `compress.sh`

Chạy trong thư mục gốc project:

```bash
cd /home/acevn/Desktop/project/orchestrator-on-edge
chmod +x compress.sh
./compress.sh
```

- Output mặc định: `../orchestrator-on-edge.tar.gz` → **`/home/acevn/Desktop/project/orchestrator-on-edge.tar.gz`**
- Tùy chọn tên: `./compress.sh my-backup` → tạo `../my-backup.tar.gz`

### Các thư mục/file được exclude (không đưa vào nén)

| Exclude | Mục đích |
|--------|----------|
| `.git` | Lịch sử git |
| `cache` | Cache các agent/orchestrator |
| `jetson-containers/data/models` | TensorRT-LLM, Hugging Face models (rất nặng) |
| `__pycache__` | Bytecode Python |
| `.venv`, `venv`, `env` | Virtualenv |
| `*.pyc` | File compiled Python |
| `voice_processing/cache` | Cache TTS/voice |

### Nén thủ công (tmux, exclude tùy chọn)

```bash
tmux new -s compress "cd /path/to/orchestrator-on-edge && tar -czvf ../orchestrator-on-edge.tar.gz --exclude='.git' --exclude='cache' --exclude='jetson-containers/data/models' --exclude='__pycache__' --exclude='voice_processing/cache' .; echo Done; exec bash"
# Detach: Ctrl+b rồi d
# Attach lại: tmux attach -t compress
```

---

## 2. Copy file nén sang máy host

Chạy lệnh **trên máy host** (máy dùng SSH vào Jetson):

```bash
# SCP
scp <user>@<jetson_ip>:/home/acevn/Desktop/project/orchestrator-on-edge.tar.gz .

# Hoặc rsync (có progress)
rsync -avz --progress <user>@<jetson_ip>:/home/acevn/Desktop/project/orchestrator-on-edge.tar.gz .
```

Thay `<user>` (vd: `acevn`) và `<jetson_ip>` bằng IP/hostname của Jetson. Đích `.` có thể đổi thành `~/Downloads/` hoặc đường dẫn khác.

---

## 3. Docker: push images lên registry

**Lưu ý:** Images build từ L4T/Jetpack và wheel aarch64 → **chỉ chạy trên ARM64 (Jetson)**. Máy x86 không chạy được.

### Bước 1: Build

```bash
cd /home/acevn/Desktop/project/orchestrator-on-edge
docker compose -p orch_v1 build
```

### Bước 2: Đăng nhập registry

```bash
docker login
# Docker Hub: username + password
# GitHub Container Registry: username + Personal Access Token
```

### Bước 3: Tag và push từng service

Thay `YOUR_USERNAME/YOUR_REPO` bằng registry của bạn (vd: `acevn/orchestrator-edge`).

```bash
# Kiểm tra tên image sau khi build
docker images | grep -E "orch_v1|voice_processing"

# Tag (tên có thể là orch_v1-<service> hoặc từ image: trong compose)
docker tag orch_v1-orchestrator:latest YOUR_USERNAME/YOUR_REPO:orchestrator
docker tag orch_v1-car_control:latest YOUR_USERNAME/YOUR_REPO:car_control
docker tag orch_v1-car_manual:latest YOUR_USERNAME/YOUR_REPO:car_manual
docker tag orch_v1-navigation:latest YOUR_USERNAME/YOUR_REPO:navigation
docker tag orch_v1-infotainment:latest YOUR_USERNAME/YOUR_REPO:infotainment
docker tag orch_v1-voice_processing:latest YOUR_USERNAME/YOUR_REPO:voice_processing
docker tag orch_v1-cloud:latest YOUR_USERNAME/YOUR_REPO:cloud
docker tag orch_v1-frontend:latest YOUR_USERNAME/YOUR_REPO:frontend

# Nếu voice_processing dùng image: voice_processing:latest
# docker tag voice_processing:latest YOUR_USERNAME/YOUR_REPO:voice_processing

# Push
docker push YOUR_USERNAME/YOUR_REPO:orchestrator
docker push YOUR_USERNAME/YOUR_REPO:car_control
docker push YOUR_USERNAME/YOUR_REPO:car_manual
docker push YOUR_USERNAME/YOUR_REPO:navigation
docker push YOUR_USERNAME/YOUR_REPO:infotainment
docker push YOUR_USERNAME/YOUR_REPO:voice_processing
docker push YOUR_USERNAME/YOUR_REPO:cloud
docker push YOUR_USERNAME/YOUR_REPO:frontend
```

---

## 4. Máy khác (Jetson): pull và chạy

### Yêu cầu

- Máy là **Jetson / ARM64**, đã cài Docker, Docker Compose, NVIDIA container runtime.
- Có **project trên máy** (clone repo hoặc giải nén `orchestrator-on-edge.tar.gz`) và file **`.env`** (copy từ máy cũ hoặc tạo mới).

### Pull images

```bash
docker pull YOUR_USERNAME/YOUR_REPO:orchestrator
docker pull YOUR_USERNAME/YOUR_REPO:car_control
docker pull YOUR_USERNAME/YOUR_REPO:car_manual
docker pull YOUR_USERNAME/YOUR_REPO:navigation
docker pull YOUR_USERNAME/YOUR_REPO:infotainment
docker pull YOUR_USERNAME/YOUR_REPO:voice_processing
docker pull YOUR_USERNAME/YOUR_REPO:cloud
docker pull YOUR_USERNAME/YOUR_REPO:frontend
```

### Chạy bằng compose dùng image

Tạo file `docker-compose.pull.yml` (hoặc sửa `docker-compose.yml`): thay mọi `build:` bằng `image: YOUR_USERNAME/YOUR_REPO:<service>`, giữ nguyên `ports`, `environment`, `volumes`. Ví dụ:

```yaml
services:
  orchestrator:
    image: YOUR_USERNAME/YOUR_REPO:orchestrator
    # ports, environment, volumes giống bản gốc
  car_control:
    image: YOUR_USERNAME/YOUR_REPO:car_control
    # ...
  voice_processing:
    image: YOUR_USERNAME/YOUR_REPO:voice_processing
    # ...
  # ... các service còn lại
```

Chạy:

```bash
docker compose -f docker-compose.pull.yml -p orch_v1 up -d
```

### Voice processing trên máy mới

Service **voice_processing** cần:

- **Audio:** PulseAudio trên host, container mount `XDG_RUNTIME_DIR/pulse`, `~/.config/pulse/cookie`, device `/dev/snd`, group `audio`.
- **network_mode: host** (đã cấu hình trong compose).
- **Thư mục:** `./voice_processing`, `./voice_processing/output`, `./voice_processing/input_test`, `./cache/voice_processing`.

Đảm bảo user chạy Docker thuộc group `audio` và Pulse đang chạy trên máy đó.

---

## 5. Tóm tắt đường dẫn & lệnh

| Nội dung | Chi tiết |
|----------|----------|
| File nén (sau khi chạy compress) | `/home/acevn/Desktop/project/orchestrator-on-edge.tar.gz` |
| Copy từ Jetson về host | Trên host: `scp user@jetson_ip:/home/acevn/Desktop/project/orchestrator-on-edge.tar.gz .` |
| Kiến trúc Docker | ARM64 (Jetson) only |
| Script nén | `./compress.sh` hoặc `./compress.sh <tên-file>` |
