# 쓰러짐 및 싸움 감지 시스템

MediaPipe Pose Estimation을 활용한 실시간 쓰러짐(Fall Detection) 및 싸움(Fight Detection) 감지 시스템입니다.

## 기능

- **쓰러짐 감지**: 포즈 랜드마크 분석을 통한 실시간 쓰러짐 감지
  - 어깨-엉덩이-발목 각도 분석
  - 키포인트 높이 비율 분석
  - 수평 위치 분석

- **싸움 감지**: 다중 인물 추적 및 움직임 패턴 분석
  - 사람들 간의 거리 분석
  - 빠른 움직임 패턴 감지
  - 쓰러짐과 연계된 싸움 감지

## 시스템 요구사항

### 하드웨어
- Jetson Orin 시리즈 (Nano, NX, AGX) 또는 일반 PC
- USB 또는 CSI 카메라 (선택사항)

### 소프트웨어
- Ubuntu 20.04 / 22.04 LTS (Jetson의 경우 JetPack 기반)
- Python 3.8 이상
- MediaPipe (Python)
- OpenCV
- Flask (웹 스트리밍 모드용)

## 설치

### 1. 시스템 업데이트 및 의존성 설치

```bash
sudo apt update && sudo apt upgrade
sudo apt install -y \
    python3-dev python3-pip python3-opencv \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev
```

### 2. Python 패키지 설치

```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

### 3. Jetson Orin 최적화 (선택사항)

Jetson Orin을 사용하는 경우 성능 최적화:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 사용법

### YOLOv11-Pose 사용 (권장 - 다중 인물 네이티브 지원)

```bash
# YOLOv11-Pose로 웹 스트리밍
python3 main.py --web --use-yolo --source 0

# YOLOv11-Pose + 최고 정확도 모델
python3 main.py --web --use-yolo --yolo-model x

# YOLOv11-Pose + 성능 최적화
python3 main.py --web --use-yolo --resize-width 480 --frame-skip 2
```

### 기본 사용 (웹캠)

```bash
python3 main.py
```

### 비디오 파일 사용

```bash
python3 main.py --source /path/to/video.mp4
```

### 출력 비디오 저장

```bash
python3 main.py --source 0 --output output.mp4
```

### SSH 환경 (화면 표시 없음)

```bash
python3 main.py --no-display --output output.mp4
```

### 웹 스트리밍 모드 (로컬 PC 브라우저에서 확인)

SSH 환경에서 실행하면서 로컬 PC의 웹 브라우저로 실시간 확인:

```bash
# 기본 포트(5000) 사용 (MediaPipe)
python3 main.py --web

# YOLOv11-Pose 사용 (다중 인물 네이티브 지원, 권장!)
python3 main.py --web --use-yolo

# YOLOv11-Pose + 최고 정확도 모델
python3 main.py --web --use-yolo --yolo-model x

# 커스텀 포트 사용
python3 main.py --web --port 8080

# 비디오 파일 스트리밍
python3 main.py --web --source /path/to/video.mp4

# MediaPipe 다중 인물 감지 (격자 분할 방식)
python3 main.py --web --multi-person

# MediaPipe 다중 인물 감지 + 커스텀 격자 (3x3)
python3 main.py --web --multi-person --grid-rows 3 --grid-cols 3
```

웹 브라우저에서 `http://서버IP:5000`으로 접속하면 실시간 스트림과 통계를 확인할 수 있습니다.

### 성능 최적화 옵션 (SSH 환경에서 FPS 향상)

```bash
# 프레임 해상도 줄이기 (640x480으로 리사이즈)
python3 main.py --web --resize-width 640

# 프레임 스킵 (2개 중 1개만 처리, 2배 빠름)
python3 main.py --web --frame-skip 2

# JPEG 품질 낮추기 (더 빠른 전송)
python3 main.py --web --jpeg-quality 50

# 포즈 그리기 비활성화 (시각화 없이 감지만)
python3 main.py --web --no-drawing

# 모든 최적화 옵션 조합 (최대 성능)
python3 main.py --web \
    --resize-width 480 \
    --frame-skip 2 \
    --jpeg-quality 60 \
    --no-drawing \
    --model-complexity 0
```

### Jetson AGX Orin 최적화 (GPU 사용)

```bash
# Jetson 클럭 최대 성능 모드
sudo nvpmodel -m 0
sudo jetson_clocks

# MediaPipe 사용 (Jetson 최적화, 권장)
python3 main.py --web \
    --source ../jungil/videos/승강기3.mp4 \
    --resize-width 480 \
    --frame-skip 2 \
    --multi-person \
    --grid-rows 2 \
    --grid-cols 2

# YOLOv11-Pose 사용 (Jetson용 PyTorch 필요)
python3 main.py --web --use-yolo \
    --source ../jungil/videos/승강기3.mp4 \
    --resize-width 320 \
    --frame-skip 5 \
    --device auto
```

### 고급 옵션

```bash
python3 main.py \
    --source 0 \
    --min-detection-confidence 0.7 \
    --min-tracking-confidence 0.7 \
    --model-complexity 2 \
    --output result.mp4
```

## 명령줄 옵션

- `--source`: 비디오 소스 (카메라 번호 또는 비디오 파일 경로, 기본값: 0)
- `--output`: 출력 비디오 파일 경로 (선택사항)
- `--min-detection-confidence`: 최소 감지 신뢰도 (기본값: 0.5)
- `--min-tracking-confidence`: 최소 추적 신뢰도 (기본값: 0.5)
- `--model-complexity`: 모델 복잡도 (0=가벼움, 1=균형, 2=무거움, 기본값: 1)
- `--no-display`: 화면 표시 비활성화 (SSH 환경용)
- `--web`: 웹 스트리밍 모드 활성화 (로컬 PC 브라우저에서 확인 가능)
- `--host`: 웹 서버 호스트 (기본값: 0.0.0.0)
- `--port`: 웹 서버 포트 (기본값: 5000)
- `--resize-width`: 프레임 리사이즈 너비 (성능 최적화, 기본값: 640, 0이면 리사이즈 안함)
- `--frame-skip`: 프레임 스킵 (N개 중 1개만 처리, 기본값: 1=모든 프레임)
- `--jpeg-quality`: JPEG 인코딩 품질 1-100 (낮을수록 빠름, 기본값: 70)
- `--no-drawing`: 포즈 그리기 비활성화 (성능 향상)
- `--multi-person`: 다중 인물 감지 활성화 (MediaPipe만, 프레임을 격자로 나눔)
- `--grid-rows`: 다중 인물 감지 격자 행 수 (기본값: 2)
- `--grid-cols`: 다중 인물 감지 격자 열 수 (기본값: 2)
- `--use-yolo`: YOLOv11-Pose 사용 (다중 인물 네이티브 지원, 권장!)
- `--yolo-model`: YOLOv11-Pose 모델 크기 (n, s, m, l, x, 기본값: n)
- `--device`: 사용할 디바이스 (cpu, cuda, 0 등, 기본값: cpu)

## 감지 알고리즘

### 쓰러짐 감지

1. **각도 분석**: 어깨-엉덩이-발목 각도가 임계값(45도) 이하일 때
2. **높이 비율**: 어깨와 엉덩이의 높이 차이가 작을 때
3. **수평 위치**: 어깨와 엉덩이가 거의 같은 높이에 있을 때

### 싸움 감지

1. **거리 분석**: 두 명 이상의 사람이 가까운 거리(150픽셀 이내)에 있을 때
2. **움직임 패턴**: 빠른 움직임(프레임 간 30픽셀 이상 이동) 감지
3. **쓰러짐 연계**: 싸움 중 쓰러짐이 감지될 때

## 성능

| 환경 | FPS | GPU 사용률 |
|------|-----|-----------|
| Jetson AGX Orin (GPU 가속) | ~25-40 FPS | 중간 |
| 일반 PC (CPU) | ~5-10 FPS | 낮음 |

## 문제 해결

### 카메라가 감지되지 않는 경우

```bash
# 카메라 테스트
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('카메라 OK' if cap.isOpened() else '카메라 오류'); cap.release()"
```

### SSH 환경에서 화면 표시 오류

```bash
export DISPLAY=:0
# 또는 --no-display 옵션 사용
# 또는 --web 옵션으로 웹 브라우저에서 확인
```

### 웹 스트리밍 접속 문제

- 방화벽에서 포트가 열려있는지 확인 (기본 포트: 5000)
- 서버 IP 주소 확인: 프로그램 실행 시 표시되는 IP 주소 사용
- 같은 네트워크에 연결되어 있는지 확인

### 낮은 FPS

- GPU 가속 활성화 (Jetson의 경우 `jetson_clocks` 실행)
- 입력 해상도 감소 (`--resize-width 480` 또는 `--resize-width 320`)
- 모델 복잡도 낮추기 (`--model-complexity 0`)
- 프레임 스킵 사용 (`--frame-skip 2` 또는 `--frame-skip 3`)
- JPEG 품질 낮추기 (`--jpeg-quality 50`)
- 포즈 그리기 비활성화 (`--no-drawing`)
- 모든 최적화 옵션 조합 사용

## 참고 자료

- [MediaPipe Pose Estimation](https://google.github.io/mediapipe/solutions/pose.html)
- [CamThink Wiki - MediaPipe 가이드](https://wiki.camthink.ai/docs/neoedge-ng4500-series/application-guide/mediapipe/)
- [NVIDIA TAO Toolkit 포즈 추정 가이드](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1/)
- [YOLOv11-Pose 문서](https://docs.ultralytics.com/tasks/pose/)

## 고급 학습 (NVIDIA TAO Toolkit)

NVIDIA TAO Toolkit을 사용하여 커스텀 포즈 추정 모델을 학습할 수 있습니다:

- **BodyPoseNet**: Bottom-up 접근 방식, 다중 인물 네이티브 지원
- **Jetson 최적화**: TensorRT 엔진 생성으로 최고 성능
- **모델 최적화**: 프루닝, 양자화 지원
- **동물 포즈 추정**: 소, 돼지 등 가축의 이상 행동 감지도 가능

자세한 내용은 다음 파일을 참고하세요:
- `TAO_TOOLKIT_GUIDE.md`: TAO Toolkit 기본 학습 가이드
- `ANIMAL_BEHAVIOR_DETECTION.md`: 동물 이상 행동 감지 가이드

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

