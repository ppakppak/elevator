# NVIDIA TAO Toolkit 포즈 추정 학습 가이드

NVIDIA TAO Toolkit을 사용한 2D 포즈 추정 모델 학습 및 최적화 가이드

## 개요

NVIDIA TAO Toolkit은 프로덕션 수준의 AI 모델을 빠르게 구축할 수 있게 해주는 도구입니다. 코드 작성 없이도 고품질 모델을 학습하고 최적화할 수 있습니다.

**참고**: [NVIDIA TAO Toolkit 포즈 추정 학습 가이드](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1/)

## TAO Toolkit의 장점

### 1. **BodyPoseNet 모델**
- Bottom-up 접근 방식 (더 빠른 추론 성능)
- 다중 인물 네이티브 지원
- 복잡한 포즈와 군중 처리에 우수
- 전역 컨텍스트 활용

### 2. **최적화 기능**
- 모델 프루닝 (Pruning)
- 양자화 (Quantization)
- TensorRT 최적화
- 실시간 추론 성능

### 3. **Jetson 최적화**
- Jetson 플랫폼에 최적화
- TensorRT 엔진 생성
- 높은 추론 성능

## BodyPoseNet 아키텍처

```
입력 이미지
    ↓
Backbone Network (특징 추출)
    ↓
Initial Prediction Stage
    ├─ Confidence Maps (Heatmap) - 키포인트 위치 예측
    └─ Part Affinity Fields (PAF) - 키포인트 연결 예측
    ↓
Multistage Refinement (0-N 단계)
    ↓
후처리 (Bipartite Matching)
    ↓
최종 포즈 (키포인트 + 연결)
```

### Bottom-up vs Top-down

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Top-down** | 사람 감지 → 각 사람의 포즈 추정 | 높은 정확도 | 느림, 사람 수에 비례 |
| **Bottom-up** | 모든 키포인트 감지 → 그룹화 | 빠름, 실시간 성능 | 구현 복잡 |

**BodyPoseNet은 Bottom-up 방식**을 사용하여 실시간 성능을 달성합니다.

## 설치 및 환경 설정

### 1. 사전 요구사항

```bash
# NVIDIA GPU (CUDA 지원)
# Docker 설치
# NGC 계정 및 API 키
```

### 2. TAO Toolkit 설치

```bash
# TAO Toolkit Launcher 설치
pip install nvidia-tao

# NGC CLI 설치
pip install nvidia-pyindex
pip install nvidia-tao
```

### 3. 샘플 다운로드

```bash
# TAO Toolkit 샘플 다운로드
ngc registry resource download-version "nvidia/tao/cv_samples:v1.2.0"
```

## 데이터 준비

### COCO 데이터셋 사용

```bash
# COCO 데이터셋 구조
data/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
│   └── *.jpg
└── val2017/
    └── *.jpg
```

### 데이터 설정 파일

```yaml
# coco_spec.json 예시
{
  "root_directory_path": "/workspace/tao-experiments/bpnet/data",
  "image_dir_name": "train2017",
  "annotations_file": "annotations/person_keypoints_train2017.json"
}
```

## 학습 설정

### 학습 스펙 파일 (bpnet_train.yaml)

```yaml
# 기본 설정
model_config:
  backbone: resnet18  # 또는 resnet50
  num_refinement_stages: 2
  input_dims: [368, 368]

train_config:
  batch_size: 16
  num_epochs: 100
  learning_rate:
    initial: 0.0001
    decay_steps: [50000, 70000]
    decay_rate: 0.1

dataset_config:
  train_dataset_path: /workspace/tao-experiments/bpnet/data
  val_dataset_path: /workspace/tao-experiments/bpnet/data
```

## 학습 실행

### 기본 학습

```bash
# 환경 변수 설정
export KEY=<your_ngc_api_key>
export NUM_GPUS=1
export LOCAL_PROJECT_DIR=/home/<username>/tao-experiments
export USER_EXPERIMENT_DIR=/workspace/tao-experiments/bpnet
export SPECS_DIR=/workspace/examples/bpnet/specs

# 학습 실행
tao bpnet train \
    -e $SPECS_DIR/bpnet_train_m1_coco.yaml \
    -r $USER_EXPERIMENT_DIR/models/exp_m1_unpruned \
    -k $KEY \
    --gpus $NUM_GPUS
```

### 다중 GPU 학습

```bash
# 4개 GPU 사용
export NUM_GPUS=4

tao bpnet train \
    -e $SPECS_DIR/bpnet_train_m1_coco.yaml \
    -r $USER_EXPERIMENT_DIR/models/exp_m1_unpruned \
    -k $KEY \
    --gpus $NUM_GPUS
```

**참고**: 다중 GPU 사용 시 학습률을 선형 스케일링해야 합니다 (learning_rate × NUM_GPUS).

## 모델 평가

### 평가 실행

```bash
# 단일 스케일 평가
tao bpnet evaluate \
    --inference_spec $SPECS_DIR/infer_spec.yaml \
    --model_filename $USER_EXPERIMENT_DIR/models/exp_m1_unpruned/bpnet_model.tlt \
    --dataset_spec $DATA_POSE_SPECS_DIR/coco_spec.json \
    --results_dir $USER_EXPERIMENT_DIR/results/exp_m1_unpruned/eval_default \
    -k $KEY
```

### 추론 테스트

```bash
# 이미지에 대한 추론
tao bpnet inference \
    --inference_spec $SPECS_DIR/infer_spec.yaml \
    --model_filename $USER_EXPERIMENT_DIR/models/exp_m1_unpruned/bpnet_model.tlt \
    --input_type dir \
    --input $USER_EXPERIMENT_DIR/data/sample_images \
    --results_dir $USER_EXPERIMENT_DIR/results/exp_m1_unpruned/infer_default \
    --dump_visualizations \
    -k $KEY
```

## 모델 최적화

### 1. 모델 프루닝 (Pruning)

```bash
# 모델 프루닝
tao bpnet prune \
    -m $USER_EXPERIMENT_DIR/models/exp_m1_unpruned/bpnet_model.tlt \
    -o $USER_EXPERIMENT_DIR/models/exp_m1_pruned/bpnet_model_pruned.tlt \
    -eq union \
    -pth 0.1 \
    -k $KEY
```

### 2. 양자화 (Quantization)

```bash
# INT8 양자화
tao bpnet calibrate \
    -m $USER_EXPERIMENT_DIR/models/exp_m1_pruned/bpnet_model_pruned.tlt \
    -o $USER_EXPERIMENT_DIR/models/exp_m1_int8/bpnet_model_int8.tlt \
    -d $DATA_DIR/val2017 \
    -k $KEY
```

### 3. TensorRT 엔진 생성

```bash
# TensorRT 엔진 생성 (Jetson 최적화)
tao bpnet export \
    -m $USER_EXPERIMENT_DIR/models/exp_m1_int8/bpnet_model_int8.tlt \
    -o $USER_EXPERIMENT_DIR/models/exp_m1_trt/bpnet_model.trt \
    -k $KEY \
    --data_type int8 \
    --batch_size 1 \
    --input_dims 368,368,3
```

## Jetson 배포

### TensorRT 엔진 사용

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

# TensorRT 엔진 로드
with open('bpnet_model.trt', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# 추론 실행
# (입력/출력 버퍼 설정 및 실행)
```

## 성능 비교

### BodyPoseNet vs 다른 모델

| 모델 | 접근 방식 | FPS (Jetson AGX) | 정확도 (mAP) |
|------|----------|------------------|--------------|
| BodyPoseNet (TAO) | Bottom-up | ~30-50 | 높음 |
| OpenPose | Bottom-up | ~10-15 | 매우 높음 |
| MediaPipe | Top-down | ~25-40 | 중간 |
| YOLOv11-Pose | Top-down | ~15-30 | 높음 |

## 현재 프로젝트와의 통합

### 옵션 1: TAO Toolkit 모델 사용

```python
# TensorRT 엔진을 사용한 추론
# (별도 구현 필요)
```

### 옵션 2: TAO Toolkit 학습 후 ONNX 변환

```bash
# ONNX로 변환
tao bpnet export \
    -m bpnet_model.tlt \
    -o bpnet_model.onnx \
    -k $KEY
```

## 학습 팁

### 1. 데이터 증강
- 회전, 스케일링, 플리핑
- 색상 조정
- 노이즈 추가

### 2. 하이퍼파라미터 튜닝
- Learning rate 스케줄링
- Batch size 조정
- Refinement stages 수 조정

### 3. 전이 학습
- 사전 학습된 백본 사용
- 도메인 특화 데이터로 fine-tuning

## 참고 자료

- [NVIDIA TAO Toolkit 공식 문서](https://docs.nvidia.com/tao/tao-toolkit/)
- [Body Pose Estimation Training 가이드](https://docs.nvidia.com/tao/tao-toolkit/text/body_pose_estimation.html)
- [TAO Toolkit GitHub](https://github.com/NVIDIA/tao-toolkit)
- [NGC TAO Toolkit](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodyposenet)

## 다음 단계

1. **Part 2**: 모델 최적화 및 배포
   - 프루닝 및 양자화
   - TensorRT 최적화
   - DeepStream 통합

2. **커스텀 데이터셋 학습**
   - 쓰러짐/싸움 감지 특화 데이터셋
   - 도메인 적응

3. **실시간 추론 파이프라인**
   - DeepStream 통합
   - GStreamer 플러그인 개발

## 현재 프로젝트 적용

현재 프로젝트에서 TAO Toolkit을 사용하려면:

1. **학습 단계**: TAO Toolkit으로 커스텀 모델 학습
2. **배포 단계**: TensorRT 엔진 생성
3. **통합 단계**: 현재 코드베이스에 TensorRT 추론 엔진 통합

TAO Toolkit은 상용 도구이므로 라이선스가 필요할 수 있습니다. 평가판 또는 교육용 라이선스를 확인하세요.

