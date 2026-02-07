# DeepStream Human Pose Estimation

NVIDIA DeepStream SDK를 활용한 실시간 Human Pose Estimation 및 이벤트 감지 시스템

## 개요

이 프로젝트는 NVIDIA DeepStream SDK와 trt_pose 모델을 사용하여 실시간으로 사람의 포즈를 추정하고, 쓰러짐/싸움 등의 이벤트를 감지합니다.

### 주요 기능

- **실시간 포즈 추정**: TensorRT 가속을 통한 고성능 추론
- **다중 인물 감지**: 동시에 여러 사람의 포즈 추정
- **이벤트 감지**:
  - 쓰러짐 감지 (Fall Detection)
  - 싸움 감지 (Fight Detection)
- **다중 입력 소스**: 카메라, 파일, RTSP 스트림 지원
- **다중 출력**: 화면, 파일, RTSP 스트리밍 지원

## 시스템 요구사항

### 하드웨어

- NVIDIA Jetson (Xavier NX, Orin 등) 또는 NVIDIA GPU가 장착된 x86 시스템
- 최소 4GB GPU 메모리 권장

### 소프트웨어

- JetPack 5.x (Jetson) 또는 CUDA 11.x (x86)
- DeepStream SDK 6.3
- TensorRT 8.x
- Python 3.8+
- GStreamer 1.0

## 설치

### 1. 의존성 설치

```bash
# DeepStream Python 바인딩
pip3 install pyds

# 기타 Python 패키지
pip3 install numpy

# trt_pose (ONNX 변환용)
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
pip3 install -e .
```

### 2. 모델 다운로드 및 변환

```bash
cd deepstream_pose

# trt_pose 모델 다운로드 및 ONNX 변환
python3 download_model.py --model resnet18_baseline
```

### 3. 커스텀 파서 빌드

```bash
# 파서 라이브러리 빌드
make

# (선택) DeepStream에 설치
sudo make install
```

## 사용법

### 기본 실행 (USB 카메라)

```bash
python3 deepstream_pose_app.py --source 0
```

### 비디오 파일 입력

```bash
python3 deepstream_pose_app.py --source file:///path/to/video.mp4
```

### RTSP 스트림 입력

```bash
python3 deepstream_pose_app.py --source rtsp://ip:port/stream
```

### 파일 출력

```bash
python3 deepstream_pose_app.py --source 0 --output output.mp4
```

### 화면 없이 실행 (SSH 환경)

```bash
python3 deepstream_pose_app.py --source 0 --no-display
```

### 전체 옵션

```bash
python3 deepstream_pose_app.py --help
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 비디오 소스 (카메라 번호 또는 URI) | 0 |
| `--output` | 출력 비디오 파일 경로 | None |
| `--no-display` | 화면 표시 비활성화 | False |
| `--rtsp` | RTSP 스트리밍 활성화 | False |
| `--rtsp-port` | RTSP 포트 | 8554 |

## 프로젝트 구조

```
deepstream_pose/
├── README.md                    # 이 파일
├── Makefile                     # 빌드 스크립트
├── download_model.py            # 모델 다운로드 스크립트
├── deepstream_pose_app.py       # 메인 애플리케이션
├── pose_visualization.py        # 시각화 유틸리티
├── config/
│   ├── pgie_pose_config.txt     # nvinfer 설정
│   ├── deepstream_pose_app.txt  # 앱 설정
│   └── labels.txt               # 레이블 파일
├── src/
│   ├── nvdsinfer_pose_parser.cpp # 커스텀 파서
│   └── pose_meta.h              # 메타데이터 헤더
├── lib/                         # 빌드된 라이브러리
│   └── libnvds_infercustomparser_pose.so
└── models/                      # 모델 파일
    ├── resnet18_baseline_224x224.onnx
    ├── resnet18_baseline_224x224.engine
    └── human_pose.json
```

## 아키텍처

### GStreamer 파이프라인

```
Source → nvstreammux → nvinfer (Pose) → nvvideoconvert → nvdsosd → Sink
                           ↓
                    Custom Parser (C++)
                           ↓
                   Pose Post-Processing
                           ↓
                   Event Detection (Python)
```

### 처리 흐름

1. **입력**: 카메라/파일/RTSP에서 비디오 스트림 수신
2. **전처리**: nvstreammux에서 배치 처리
3. **추론**: nvinfer에서 trt_pose 모델 실행
4. **후처리**: 커스텀 파서에서 키포인트 추출
5. **분석**: Python에서 쓰러짐/싸움 감지
6. **시각화**: nvdsosd에서 스켈레톤 및 정보 표시
7. **출력**: 화면/파일/RTSP로 결과 전송

## 성능

### Jetson Xavier NX

| 모델 | 해상도 | FPS (1 스트림) | FPS (3 스트림) |
|------|--------|----------------|----------------|
| ResNet18 | 224x224 | ~20 | ~10 |
| DenseNet121 | 256x256 | ~15 | ~7 |

### Jetson Orin

| 모델 | 해상도 | FPS (1 스트림) | FPS (4 스트림) |
|------|--------|----------------|----------------|
| ResNet18 | 224x224 | ~40 | ~20 |
| DenseNet121 | 256x256 | ~30 | ~15 |

## 이벤트 감지 알고리즘

### 쓰러짐 감지

다음 조건들을 종합하여 쓰러짐 여부 판단:

1. **높이 비율**: 어깨와 엉덩이의 높이 차이
2. **수평 위치**: 어깨와 엉덩이가 수평에 가까운지
3. **각도 분석**: 어깨-엉덩이-발목 각도

### 싸움 감지

다음 조건들을 종합하여 싸움 여부 판단:

1. **근접도**: 두 사람 간의 거리
2. **움직임**: 프레임 간 위치 변화
3. **연계 이벤트**: 쓰러짐과의 연계

## 문제 해결

### 빌드 오류

```bash
# DeepStream 경로 확인
ls /opt/nvidia/deepstream/deepstream-6.3/

# CUDA 경로 확인
ls /usr/local/cuda/
```

### 실행 오류

```bash
# GStreamer 플러그인 확인
gst-inspect-1.0 nvinfer

# 권한 확인 (카메라 접근)
sudo chmod 666 /dev/video0
```

### 성능 최적화

1. **FP16 모드 사용**: config에서 `network-mode=2`
2. **배치 크기 증가**: 다중 스트림 시
3. **해상도 조정**: 입력 해상도 낮추기

## 참고 자료

- [NVIDIA DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [trt_pose GitHub](https://github.com/NVIDIA-AI-IOT/trt_pose)
- [DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
- [Original NVIDIA Blog Post](https://developer.nvidia.com/blog/creating-a-human-pose-estimation-application-with-deepstream-sdk/)

## 라이센스

MIT License
