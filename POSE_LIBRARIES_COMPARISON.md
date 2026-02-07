# 포즈 추정 라이브러리 비교 가이드

쓰러짐 및 싸움 감지를 위한 포즈 추정 라이브러리와 프레임워크 비교

## 주요 포즈 추정 라이브러리

### 0. **NVIDIA TAO Toolkit BodyPoseNet** ⭐ 최고 성능 (상용)
- **개발사**: NVIDIA
- **다중 인물**: 네이티브 지원 (Bottom-up)
- **정확도**: 매우 높음
- **속도**: 매우 빠름 (Jetson 최적화)
- **설치 난이도**: 어려움 (상용 라이선스 필요)
- **장점**:
  - Jetson에 최적화된 성능
  - TensorRT 엔진 지원
  - 모델 프루닝 및 양자화
  - Bottom-up 접근 (실시간 성능)
  - 다중 인물 네이티브 지원
- **단점**:
  - 상용 라이선스 필요
  - 설치 및 설정 복잡
  - NGC 계정 필요
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐⭐ (최고)
- **참고**: [NVIDIA TAO Toolkit 가이드](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1/)

### 1. **MediaPipe Pose** (현재 사용 중)
- **개발사**: Google
- **다중 인물**: 기본적으로 단일 인물 (격자 분할로 다중 감지 가능)
- **정확도**: 중간~높음
- **속도**: 매우 빠름 (실시간)
- **설치 난이도**: 쉬움
- **장점**:
  - 설치 및 사용이 매우 간단
  - 실시간 성능 우수
  - 경량 모델
  - Python API 제공
- **단점**:
  - 기본적으로 한 명만 감지
  - 정확도가 OpenPose나 AlphaPose보다 낮을 수 있음
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐ (좋음)

### 2. **OpenPose**
- **개발사**: CMU (Carnegie Mellon University)
- **다중 인물**: 네이티브 지원 (최대 15명)
- **정확도**: 매우 높음
- **속도**: 중간 (GPU 필요)
- **설치 난이도**: 어려움
- **장점**:
  - 다중 인물 네이티브 지원
  - 매우 정확한 키포인트 감지
  - 손, 얼굴, 발까지 감지 가능
  - 오픈소스
- **단점**:
  - 설치가 복잡 (Caffe, CUDA 필요)
  - GPU 필수
  - 실시간 성능이 MediaPipe보다 느림
  - Python 래퍼 사용 필요
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐⭐ (매우 좋음)
- **설치**: 
  ```bash
  # OpenPose Python API
  pip install openpose-python
  # 또는 직접 빌드 (복잡함)
  ```

### 3. **YOLOv8-Pose**
- **개발사**: Ultralytics
- **다중 인물**: 네이티브 지원
- **정확도**: 높음
- **속도**: 빠름
- **설치 난이도**: 쉬움
- **장점**:
  - 객체 감지와 포즈 추정 통합
  - 다중 인물 자동 감지
  - 설치 및 사용이 간단
  - GPU/CPU 모두 지원
  - 실시간 성능 우수
- **단점**:
  - 키포인트 수가 적을 수 있음 (17개)
  - MediaPipe보다 정확도가 약간 낮을 수 있음
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐⭐ (매우 좋음)
- **설치**:
  ```bash
  pip install ultralytics
  ```

### 3-1. **YOLOv11-Pose** ⭐ 최신 버전
- **개발사**: Ultralytics
- **다중 인물**: 네이티브 지원
- **정확도**: 매우 높음 (YOLOv8보다 향상)
- **속도**: 빠름
- **설치 난이도**: 쉬움
- **장점**:
  - YOLOv8의 개선된 버전
  - 더 높은 정확도 (mAP 50.0~69.5)
  - 객체 감지와 포즈 추정 통합
  - 다중 인물 자동 감지
  - 설치 및 사용이 간단
  - GPU/CPU 모두 지원
  - 실시간 성능 우수
  - 다양한 모델 크기 제공 (n, s, m, l, x)
- **단점**:
  - 키포인트 수가 적을 수 있음 (17개)
  - MediaPipe보다 키포인트 수는 적지만 정확도는 높음
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐⭐ (매우 좋음 - 최고 추천!)
- **성능**:
  - YOLOv11n-pose: mAP 50.0 (가장 빠름)
  - YOLOv11s-pose: mAP ~55-60
  - YOLOv11m-pose: mAP ~62-65
  - YOLOv11l-pose: mAP ~66-68
  - YOLOv11x-pose: mAP 69.5 (가장 정확)
- **설치**:
  ```bash
  pip install ultralytics
  ```

### 4. **AlphaPose**
- **개발사**: ShanghaiTech University
- **다중 인물**: 네이티브 지원
- **정확도**: 매우 높음
- **속도**: 중간
- **설치 난이도**: 중간
- **장점**:
  - 다중 인물 감지 정확도 높음
  - 다양한 환경에서 안정적
  - 오픈소스
- **단점**:
  - 설치가 복잡할 수 있음
  - 실시간 성능이 MediaPipe보다 느림
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐ (좋음)
- **설치**:
  ```bash
  pip install alphapose
  ```

### 5. **MMPose** (OpenMMLab)
- **개발사**: OpenMMLab
- **다중 인물**: 네이티브 지원
- **정확도**: 매우 높음
- **속도**: 중간~빠름
- **설치 난이도**: 중간
- **장점**:
  - 다양한 모델 아키텍처 지원
  - 높은 정확도
  - PyTorch 기반
  - 모듈화된 구조
- **단점**:
  - 설정이 복잡할 수 있음
  - GPU 권장
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐⭐ (좋음)
- **설치**:
  ```bash
  pip install mmpose
  ```

### 6. **MoveNet** (TensorFlow)
- **개발사**: Google
- **다중 인물**: 단일 인물 (Lightning) / 다중 인물 (Thunder)
- **정확도**: 높음
- **속도**: 매우 빠름
- **설치 난이도**: 쉬움
- **장점**:
  - 매우 빠른 속도
  - TensorFlow 통합
  - 모바일 최적화
- **단점**:
  - 다중 인물 모델(Thunder)은 무거움
  - 키포인트 수가 적을 수 있음
- **쓰러짐/싸움 감지 적합성**: ⭐⭐⭐ (보통)
- **설치**:
  ```bash
  pip install tensorflow
  ```

### 7. **PoseNet** (TensorFlow.js)
- **개발사**: Google
- **다중 인물**: 단일 인물
- **정확도**: 중간
- **속도**: 빠름
- **설치 난이도**: 쉬움
- **장점**:
  - 웹 브라우저에서 실행 가능
  - 경량 모델
- **단점**:
  - 단일 인물만 지원
  - 정확도가 낮음
- **쓰러짐/싸움 감지 적합성**: ⭐⭐ (낮음)

## 비교표

| 라이브러리 | 다중 인물 | 정확도 | 속도 | 설치 | GPU 필요 | 쓰러짐/싸움 |
|-----------|---------|--------|------|------|----------|------------|
| **TAO BodyPoseNet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 권장 | ⭐⭐⭐⭐⭐ |
| MediaPipe | ⚠️ (격자 분할) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 선택 | ⭐⭐⭐⭐ |
| **YOLOv11-Pose** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 선택 | ⭐⭐⭐⭐⭐ |
| YOLOv8-Pose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 선택 | ⭐⭐⭐⭐⭐ |
| OpenPose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 필수 | ⭐⭐⭐⭐⭐ |
| AlphaPose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 권장 | ⭐⭐⭐⭐ |
| MMPose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 권장 | ⭐⭐⭐⭐ |
| MoveNet | ⚠️ (Thunder) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 선택 | ⭐⭐⭐ |
| PoseNet | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 선택 | ⭐⭐ |

## 추천 라이브러리

### 🏆 최고 추천: **YOLOv11-Pose** (2024 최신)
- **싸움 감지에 최적**: 다중 인물 네이티브 지원 + 높은 정확도
- **쓰러짐 감지에 최적**: 향상된 정확도 (mAP 69.5)
- **실시간 성능**: 빠른 처리 속도
- **설치 용이**: `pip install ultralytics` 한 줄로 설치
- **다양한 모델 크기**: n(빠름) ~ x(정확) 선택 가능

### 싸움 감지에 최적: **YOLOv11-Pose** 또는 **YOLOv8-Pose**
- 다중 인물 네이티브 지원
- 높은 정확도
- 실시간 성능
- 설치가 쉬움

### 쓰러짐 감지에 최적: **YOLOv11-Pose** 또는 **OpenPose**
- 매우 정확한 키포인트 감지
- 다중 인물 지원
- YOLOv11-Pose는 설치가 더 쉬움

### 균형잡힌 선택: **YOLOv11-Pose**
- 최신 기술
- 설치가 쉬움
- 다중 인물 지원
- 최고 성능
- 실시간 처리 가능

## YOLOv11-Pose 통합 예제 (권장)

YOLOv11-Pose는 싸움 감지에 가장 적합한 최신 선택입니다:

```python
from ultralytics import YOLO

# YOLOv11-Pose 모델 로드
model = YOLO('yolov11n-pose.pt')  # nano (가장 빠름, mAP 50.0)
# model = YOLO('yolov11s-pose.pt')  # small (균형)
# model = YOLO('yolov11m-pose.pt')  # medium
# model = YOLO('yolov11l-pose.pt')  # large
# model = YOLO('yolov11x-pose.pt')  # xlarge (가장 정확, mAP 69.5)

# 비디오 처리
results = model('video.mp4', stream=True)

for result in results:
    # 다중 인물 포즈 감지
    keypoints = result.keypoints  # [num_people, 17, 3]
    boxes = result.boxes  # 바운딩 박스
    
    # 각 사람의 키포인트 처리
    for person_idx in range(len(keypoints)):
        person_keypoints = keypoints[person_idx]
        # 쓰러짐/싸움 감지 로직 적용
```

## YOLOv8-Pose 통합 예제 (이전 버전)

YOLOv8-Pose도 여전히 사용 가능합니다:

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n-pose.pt')  # nano (가장 빠름)
# model = YOLO('yolov8s-pose.pt')  # small
# model = YOLO('yolov8m-pose.pt')  # medium
# model = YOLO('yolov8l-pose.pt')  # large
# model = YOLO('yolov8x-pose.pt')  # xlarge (가장 정확)

# 비디오 처리
results = model('video.mp4', stream=True)

for result in results:
    # 다중 인물 포즈 감지
    keypoints = result.keypoints  # [num_people, 17, 3]
    boxes = result.boxes  # 바운딩 박스
    
    # 각 사람의 키포인트 처리
    for person_idx in range(len(keypoints)):
        person_keypoints = keypoints[person_idx]
        # 쓰러짐/싸움 감지 로직 적용
```

## OpenPose 통합 예제

OpenPose는 가장 정확하지만 설치가 복잡합니다:

```python
import cv2
from openpose import pyopenpose as op

# OpenPose 설정
params = {
    "model_folder": "models/",
    "number_people_max": 15,  # 최대 15명
    "net_resolution": "320x176"  # 성능 최적화
}

opWrapper = op.Wrapper()
opWrapper.configure(params)
opWrapper.start()

# 프레임 처리
datum = op.Datum()
datum.cvInputData = frame
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# 다중 인물 포즈
poses = datum.poseKeypoints  # [num_people, 25, 3]
```

## 마이그레이션 고려사항

### MediaPipe → YOLOv11-Pose (최신, 강력 추천!)
- ✅ 설치가 쉬움 (`pip install ultralytics`)
- ✅ 다중 인물 네이티브 지원
- ✅ 성능 향상 (YOLOv8보다 더 정확)
- ✅ 최신 기술
- ⚠️ 키포인트 인덱스가 다름 (17개 vs MediaPipe 33개)
- ⚠️ 모델 파일 자동 다운로드 (처음 실행 시)

### MediaPipe → YOLOv8-Pose
- ✅ 설치가 쉬움
- ✅ 다중 인물 네이티브 지원
- ✅ 성능 향상 가능
- ⚠️ 키포인트 인덱스가 다름 (17개 vs 33개)
- ⚠️ YOLOv11보다 정확도가 약간 낮음

### MediaPipe → OpenPose
- ✅ 정확도 향상
- ✅ 다중 인물 네이티브 지원
- ⚠️ 설치 복잡
- ⚠️ GPU 필수
- ⚠️ 성능 저하 가능

## 결론

**현재 상황 (MediaPipe 사용 중)**:
- 싸움 감지를 위해 다중 인물 감지가 필요하다면 **YOLOv11-Pose**로 마이그레이션 강력 권장
- 정확도가 가장 중요하다면 **YOLOv11-Pose** 또는 **OpenPose** 고려
- 현재 시스템이 잘 작동한다면 MediaPipe + 격자 분할 방식 유지 가능

**권장 순서**:
1. **TAO BodyPoseNet** ⭐⭐⭐ - 최고 성능 (Jetson 최적화, 상용 라이선스 필요)
2. **YOLOv11-Pose** ⭐⭐ - 최신 기술, 최고 성능, 쉬운 설치
3. **YOLOv8-Pose** - 안정적인 선택
4. **OpenPose** - 최고 정확도 필요 시 (설치 복잡)
5. **MediaPipe (현재)** - 현재 시스템 유지

**YOLOv11-Pose 선택 이유**:
- ✅ 2024년 최신 버전
- ✅ YOLOv8보다 향상된 정확도 (mAP 69.5)
- ✅ 다중 인물 네이티브 지원
- ✅ 설치가 매우 쉬움
- ✅ 실시간 성능 우수
- ✅ 다양한 모델 크기 선택 가능

