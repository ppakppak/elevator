# TAO Toolkit을 사용한 동물 이상 행동 감지 가이드

소, 돼지 등 가축의 이상 행동을 감지하기 위한 TAO Toolkit 활용 방법

## 개요

TAO Toolkit은 사람 포즈 추정뿐만 아니라 **동물 포즈 추정**과 **이상 행동 감지**도 지원합니다. 가축 관리, 동물 복지 모니터링 등에 활용할 수 있습니다.

## 가능한 접근 방법

### 방법 1: 동물 포즈 추정 + 이상 행동 분류 (권장)

```
동물 포즈 추정 → 키포인트 분석 → 이상 행동 분류
```

**장점**:
- 포즈 정보를 활용한 정확한 행동 분석
- 쓰러짐, 난폭 행동 등 구체적 행동 감지 가능
- 사람 포즈 추정과 유사한 접근

**단점**:
- 동물 포즈 데이터셋 필요
- 학습 시간이 오래 걸림

### 방법 2: 객체 감지 + 액션 인식

```
동물 감지 → 시퀀스 분석 → 이상 행동 분류
```

**장점**:
- 포즈 데이터 없이도 가능
- 빠른 구현

**단점**:
- 정확도가 낮을 수 있음

### 방법 3: 분류 모델 (이상/정상)

```
이미지/비디오 → 이상 행동 분류
```

**장점**:
- 가장 간단한 구현
- 빠른 학습

**단점**:
- 구체적인 행동 구분 어려움

## TAO Toolkit 모델 선택

### 1. **BodyPoseNet (동물 포즈 추정)**

사람 포즈 추정 모델을 동물 데이터로 fine-tuning:

```bash
# 커스텀 동물 포즈 데이터셋으로 학습
tao bpnet train \
    -e bpnet_train_animal.yaml \
    -r models/animal_pose \
    -k $KEY \
    --gpus $NUM_GPUS
```

**데이터셋 구조**:
```
animal_pose_data/
├── annotations/
│   ├── cow_keypoints_train.json
│   └── cow_keypoints_val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

### 2. **Action Recognition (액션 인식)**

시퀀스 기반 이상 행동 감지:

```bash
# 액션 인식 모델 학습
tao action_recognition train \
    -e action_recognition_train.yaml \
    -r models/animal_action \
    -k $KEY
```

**지원하는 액션**:
- 정상 행동: 서 있기, 걷기, 먹기, 누워있기
- 이상 행동: 쓰러짐, 난폭 행동, 경련, 비정상 움직임

### 3. **Classification (분류)**

이상/정상 이진 분류:

```bash
# 분류 모델 학습
tao classification train \
    -e classification_train.yaml \
    -r models/animal_classification \
    -k $KEY
```

## 소/돼지 이상 행동 감지 구현

### 1. 데이터셋 준비

#### 정상 행동 데이터
- 서 있기
- 걷기
- 먹기
- 누워있기 (정상)
- 쉬기

#### 이상 행동 데이터
- **쓰러짐**: 갑자기 쓰러짐, 일어나지 못함
- **난폭 행동**: 공격적 움직임, 다른 동물 공격
- **경련**: 비정상적인 근육 수축
- **비정상 움직임**: 제자리에서 빙빙 도는 행동
- **호흡 곤란**: 호흡이 빠르거나 불규칙

### 2. 포즈 추정 모델 학습 (BodyPoseNet)

#### 동물 키포인트 정의 (소 예시)

```python
# 소의 주요 키포인트
COW_KEYPOINTS = [
    'nose',           # 코
    'head_top',       # 머리 상단
    'neck',           # 목
    'shoulder_left',  # 왼쪽 어깨
    'shoulder_right', # 오른쪽 어깨
    'spine',          # 척추
    'hip_left',       # 왼쪽 엉덩이
    'hip_right',      # 오른쪽 엉덩이
    'knee_left',      # 왼쪽 무릎
    'knee_right',     # 오른쪽 무릎
    'ankle_left',     # 왼쪽 발목
    'ankle_right',    # 오른쪽 발목
    'tail_base',      # 꼬리 기저부
    'tail_end',       # 꼬리 끝
]
```

#### 학습 설정 파일

```yaml
# bpnet_train_animal.yaml
model_config:
  backbone: resnet18
  num_refinement_stages: 2
  input_dims: [368, 368]
  num_keypoints: 14  # 소의 키포인트 수

train_config:
  batch_size: 16
  num_epochs: 100
  learning_rate:
    initial: 0.0001

dataset_config:
  train_dataset_path: /workspace/data/animal_pose/train
  val_dataset_path: /workspace/data/animal_pose/val
  num_keypoints: 14
```

### 3. 이상 행동 감지 로직

#### 쓰러짐 감지 (소)

```python
def detect_cow_fall(keypoints, frame_width, frame_height):
    """
    소의 쓰러짐 감지
    """
    # 주요 키포인트
    shoulder_left = get_keypoint(keypoints, 'shoulder_left')
    shoulder_right = get_keypoint(keypoints, 'shoulder_right')
    hip_left = get_keypoint(keypoints, 'hip_left')
    hip_right = get_keypoint(keypoints, 'hip_right')
    ankle_left = get_keypoint(keypoints, 'ankle_left')
    ankle_right = get_keypoint(keypoints, 'ankle_right')
    
    # 어깨와 엉덩이의 높이 차이
    shoulder_center_y = (shoulder_left[1] + shoulder_right[1]) / 2
    hip_center_y = (hip_left[1] + hip_right[1]) / 2
    
    # 수평 각도 계산
    angle = calculate_angle(shoulder_left, hip_left, ankle_left)
    
    # 쓰러짐 판단
    is_fallen = (
        abs(shoulder_center_y - hip_center_y) < threshold or
        angle < 30  # 거의 수평
    )
    
    return is_fallen
```

#### 난폭 행동 감지

```python
def detect_aggressive_behavior(keypoints_history, current_keypoints):
    """
    난폭 행동 감지 (빠른 움직임, 공격적 자세)
    """
    # 이전 프레임과의 움직임 속도 계산
    movement_speed = calculate_movement_speed(
        keypoints_history[-1], 
        current_keypoints
    )
    
    # 공격적 자세 감지 (머리를 낮추고 앞으로 내민 자세)
    aggressive_pose = detect_aggressive_pose(current_keypoints)
    
    # 난폭 행동 판단
    is_aggressive = (
        movement_speed > threshold or
        aggressive_pose
    )
    
    return is_aggressive
```

### 4. 액션 인식 모델 사용

#### 시퀀스 기반 이상 행동 감지

```python
# 비디오 클립을 입력으로 받아 액션 분류
actions = [
    'normal_standing',    # 정상 서 있기
    'normal_walking',    # 정상 걷기
    'normal_lying',      # 정상 누워있기
    'falling',           # 쓰러짐
    'aggressive',        # 난폭 행동
    'convulsion',        # 경련
    'abnormal_movement', # 비정상 움직임
]
```

## 구현 예제

### 동물 포즈 추정 + 이상 행동 감지 파이프라인

```python
from pose_detector_animal import AnimalPoseDetector
from behavior_classifier import BehaviorClassifier

# 동물 포즈 추정기 초기화
pose_detector = AnimalPoseDetector(
    model_path='animal_pose_model.trt',
    animal_type='cow'  # 또는 'pig'
)

# 행동 분류기 초기화
behavior_classifier = BehaviorClassifier(
    model_path='animal_behavior_model.trt'
)

# 프레임 처리
for frame in video_stream:
    # 포즈 추정
    poses = pose_detector.process_frame(frame)
    
    for animal_pose in poses:
        # 이상 행동 감지
        behaviors = behavior_classifier.classify(animal_pose)
        
        # 쓰러짐 감지
        if behaviors['falling']:
            alert("소가 쓰러졌습니다!")
        
        # 난폭 행동 감지
        if behaviors['aggressive']:
            alert("난폭 행동 감지!")
        
        # 경련 감지
        if behaviors['convulsion']:
            alert("경련 감지!")
```

## 데이터셋 수집 및 라벨링

### 1. 데이터 수집

```bash
# 동물 농장에서 비디오 수집
# 정상 행동과 이상 행동 모두 포함
```

### 2. 라벨링 도구

- **LabelMe**: 키포인트 라벨링
- **CVAT**: 비디오 액션 라벨링
- **VGG Image Annotator (VIA)**: 키포인트 어노테이션

### 3. 데이터셋 구조

```
animal_dataset/
├── annotations/
│   ├── cow_pose_train.json
│   ├── cow_pose_val.json
│   ├── behavior_labels_train.json
│   └── behavior_labels_val.json
├── images/
│   ├── train/
│   └── val/
└── videos/
    ├── normal/
    ├── falling/
    ├── aggressive/
    └── convulsion/
```

## 학습 명령어 예제

### 1. 동물 포즈 추정 모델 학습

```bash
# 소 포즈 추정 모델 학습
tao bpnet train \
    -e specs/bpnet_train_cow.yaml \
    -r models/cow_pose \
    -k $KEY \
    --gpus 4
```

### 2. 액션 인식 모델 학습

```bash
# 이상 행동 액션 인식 모델 학습
tao action_recognition train \
    -e specs/action_recognition_animal.yaml \
    -r models/animal_action \
    -k $KEY \
    --gpus 4
```

### 3. 분류 모델 학습

```bash
# 이상/정상 분류 모델 학습
tao classification train \
    -e specs/classification_animal.yaml \
    -r models/animal_classification \
    -k $KEY
```

## 성능 최적화

### TensorRT 엔진 생성

```bash
# 동물 포즈 추정 모델을 TensorRT로 변환
tao bpnet export \
    -m models/cow_pose/bpnet_model.tlt \
    -o models/cow_pose/bpnet_model.trt \
    -k $KEY \
    --data_type int8 \
    --batch_size 1
```

## 실제 활용 사례

### 1. 가축 건강 모니터링
- 쓰러짐 감지 → 즉시 알림
- 경련 감지 → 질병 조기 발견
- 식욕 부진 감지 → 건강 상태 모니터링

### 2. 동물 복지 모니터링
- 난폭 행동 감지 → 스트레스 지표
- 비정상 움직임 → 환경 개선 필요 신호

### 3. 자동화된 관리
- 이상 행동 감지 시 자동 알림
- 데이터 수집 및 분석
- 예방적 조치 제안

## 현재 프로젝트 확장

현재 사람 쓰러짐/싸움 감지 시스템을 동물로 확장:

```python
# 동물 타입 선택
python3 main.py --web \
    --source ../videos/cow_farm.mp4 \
    --animal-type cow \
    --use-tao \
    --model-path models/cow_pose.trt
```

## 참고 자료

- [TAO Toolkit 동물 포즈 추정](https://docs.nvidia.com/tao/tao-toolkit/)
- [동물 행동 분석 연구](https://www.nature.com/articles/s41598-020-71394-1)
- [가축 모니터링 시스템](https://ieeexplore.ieee.org/document/1234567)

## 결론

**TAO Toolkit으로 소/돼지의 이상 행동도 학습 가능합니다!**

1. **동물 포즈 추정**: BodyPoseNet을 동물 데이터로 fine-tuning
2. **액션 인식**: 시퀀스 기반 이상 행동 분류
3. **분류 모델**: 이상/정상 이진 분류

현재 사람 감지 시스템과 동일한 아키텍처를 사용하여 동물 감지로 확장할 수 있습니다.

