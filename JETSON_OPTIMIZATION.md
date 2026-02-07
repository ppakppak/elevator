# Jetson AGX Orin 성능 최적화 가이드

## 현재 문제

- PyTorch가 CUDA를 인식하지 못함 (CPU만 사용)
- 1 FPS로 매우 느린 성능
- Jetson용 PyTorch가 설치되지 않았을 가능성

## 해결 방법

### 1. Jetson용 PyTorch 설치 (권장)

Jetson AGX Orin에서는 Jetson용 PyTorch를 설치해야 합니다:

```bash
# JetPack 버전 확인
cat /etc/nv_tegra_release

# Jetson용 PyTorch 설치 (JetPack 5.x 기준)
# 공식 NVIDIA 가이드 참조: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

### 2. MediaPipe 사용 (즉시 사용 가능)

MediaPipe는 Jetson에서 더 잘 최적화되어 있습니다:

```bash
# MediaPipe 사용 (GPU 가속 지원)
python3 main.py --web --source ../jungil/videos/승강기3.mp4 \
    --resize-width 480 \
    --frame-skip 3 \
    --multi-person \
    --grid-rows 2 \
    --grid-cols 2
```

### 3. 성능 최적화 옵션 (YOLOv11-Pose 사용 시)

```bash
# 최대 성능 모드
python3 main.py --web --use-yolo \
    --source ../jungil/videos/승강기3.mp4 \
    --resize-width 320 \
    --frame-skip 5 \
    --jpeg-quality 50 \
    --no-drawing \
    --yolo-model n
```

### 4. Jetson 클럭 최적화

```bash
# 최대 성능 모드로 설정
sudo nvpmodel -m 0
sudo jetson_clocks

# 확인
sudo jetson_clocks --show
```

## 성능 비교

| 설정 | 예상 FPS (CPU) | 예상 FPS (GPU) |
|------|---------------|---------------|
| MediaPipe (최적화) | 5-10 | 25-40 |
| YOLOv11-Pose (CPU) | 1-3 | - |
| YOLOv11-Pose (GPU) | - | 15-30 |

## 권장 설정

### 즉시 사용 가능 (MediaPipe)

```bash
python3 main.py --web \
    --source ../jungil/videos/승강기3.mp4 \
    --resize-width 480 \
    --frame-skip 2 \
    --multi-person \
    --grid-rows 2 \
    --grid-cols 2
```

### 최고 성능 (MediaPipe)

```bash
python3 main.py --web \
    --source ../jungil/videos/승강기3.mp4 \
    --resize-width 320 \
    --frame-skip 3 \
    --multi-person \
    --grid-rows 2 \
    --grid-cols 2 \
    --no-drawing
```

## Jetson PyTorch 설치 확인

```bash
# PyTorch CUDA 지원 확인
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Version:', torch.__version__)"

# Jetson 정보 확인
cat /etc/nv_tegra_release
```

## 참고

- MediaPipe는 Jetson에서 TensorRT와 통합되어 더 빠름
- YOLOv11-Pose는 Jetson용 PyTorch가 필요함
- 현재는 MediaPipe 사용을 권장합니다

