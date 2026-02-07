# YOLOv11-Pose 사용 가이드

## 설치

```bash
source venv/bin/activate
pip install ultralytics
```

## 기본 사용

### 웹 스트리밍 모드 (권장)

```bash
# YOLOv11-Pose 기본 사용
python3 main.py --web --use-yolo --source 0

# 비디오 파일 처리
python3 main.py --web --use-yolo --source ../jungil/videos/승강기1.mp4
```

### 모델 크기 선택

```bash
# 빠른 모델 (nano) - 기본값
python3 main.py --web --use-yolo --yolo-model n

# 균형잡힌 모델 (small)
python3 main.py --web --use-yolo --yolo-model s

# 정확한 모델 (xlarge) - 가장 정확
python3 main.py --web --use-yolo --yolo-model x
```

## 성능 최적화

```bash
# YOLOv11-Pose + 성능 최적화
python3 main.py --web --use-yolo \
    --resize-width 480 \
    --frame-skip 2 \
    --jpeg-quality 60
```

## MediaPipe vs YOLOv11-Pose

| 기능 | MediaPipe | YOLOv11-Pose |
|------|-----------|--------------|
| 다중 인물 | 격자 분할 | 네이티브 지원 |
| 정확도 | 중간 | 높음 |
| 속도 | 빠름 | 빠름 |

## 장점

- ✅ 다중 인물 네이티브 지원 (격자 분할 불필요)
- ✅ 높은 정확도 (mAP 69.5)
- ✅ 실시간 성능 우수
- ✅ 설치 쉬움

