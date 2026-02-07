"""
YOLO11 Instance Segmentation 기반 낙상/싸움 감지 모듈
학습된 세그멘테이션 모델을 사용하여 직접 상태 분류
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """감지 결과"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: Optional[np.ndarray]        # 세그멘테이션 마스크
    class_id: int                     # 0=정상, 1=쓰러짐, 2=싸움
    class_name: str
    confidence: float

    @property
    def is_fallen(self) -> bool:
        return self.class_id == 1

    @property
    def is_fighting(self) -> bool:
        return self.class_id == 2

    @property
    def is_normal(self) -> bool:
        return self.class_id == 0


class SegmentationDetector:
    """
    YOLO11 Instance Segmentation 기반 감지기

    클래스:
        0: normal (정상)
        1: deformation (쓰러짐)
        2: crack (싸움)
    """

    # 클래스 이름 매핑 (한글)
    CLASS_NAMES_KR = {
        0: '정상',
        1: '쓰러짐',
        2: '싸움'
    }

    # 시각화 색상 (BGR)
    CLASS_COLORS = {
        0: (0, 255, 0),    # 녹색 - 정상
        1: (0, 0, 255),    # 빨강 - 쓰러짐
        2: (0, 165, 255)   # 주황 - 싸움
    }

    def __init__(self,
                 model_path: str = 'models/best_m.pt',
                 confidence: float = 0.3,
                 imgsz: int = 384,  # 384 권장 (21.8 FPS, 정확도 87%)
                 device: str = 'auto'):
        """
        Args:
            model_path: 세그멘테이션 모델 경로
            confidence: 최소 신뢰도
            imgsz: 추론 이미지 크기
            device: 디바이스 (auto, cpu, cuda, 0, etc.)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz

        # 디바이스 설정
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # TensorRT 엔진 자동 감지
        engine_path = model_path.replace('.pt', '.engine')
        if os.path.exists(engine_path) and self.device != 'cpu':
            print(f"TensorRT 엔진 사용: {engine_path}")
            self.model = YOLO(engine_path, task='segment')
            self.use_tensorrt = True
        else:
            print(f"세그멘테이션 모델 로딩: {model_path}")
            self.model = YOLO(model_path)
            self.use_tensorrt = False
        print(f"모델 로드 완료 (디바이스: {self.device}, TensorRT: {self.use_tensorrt})")

        # 클래스 이름 확인
        self.class_names = self.model.names
        print(f"클래스: {self.class_names}")

        # 프레임 카운터
        self.frame_count = 0

        # 워밍업
        self._warmup()

    def _warmup(self):
        """모델 워밍업"""
        print("모델 워밍업 중...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            if self.use_tensorrt:
                self.model(dummy, verbose=False)
            else:
                self.model(dummy, verbose=False, imgsz=self.imgsz)
        print("워밍업 완료")

    def process_frame(self, frame: np.ndarray,
                      frame_number: int = None) -> List[DetectionResult]:
        """
        프레임 처리 및 감지

        Args:
            frame: BGR 이미지
            frame_number: 프레임 번호 (선택)

        Returns:
            DetectionResult 리스트
        """
        if frame_number is not None:
            self.frame_count = frame_number
        else:
            self.frame_count += 1

        # 추론 (TensorRT는 빌드 시 설정 고정)
        if self.use_tensorrt:
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False,
                imgsz=self.imgsz,
                device=self.device
            )

        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks

            for i, box in enumerate(boxes):
                # 바운딩 박스
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # 클래스 및 신뢰도
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.CLASS_NAMES_KR.get(class_id,
                                                     self.class_names.get(class_id, 'unknown'))

                # 마스크 (있는 경우)
                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy()[0]

                detections.append(DetectionResult(
                    bbox=bbox,
                    mask=mask,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence
                ))

        return detections

    def draw_results(self, frame: np.ndarray,
                     detections: List[DetectionResult],
                     show_mask: bool = True,
                     show_bbox: bool = True,
                     show_label: bool = True,
                     mask_alpha: float = 0.4) -> np.ndarray:
        """
        감지 결과 시각화

        Args:
            frame: 원본 프레임
            detections: 감지 결과 리스트
            show_mask: 마스크 표시 여부
            show_bbox: 바운딩박스 표시 여부
            show_label: 레이블 표시 여부
            mask_alpha: 마스크 투명도

        Returns:
            시각화된 프레임
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        for det in detections:
            color = self.CLASS_COLORS.get(det.class_id, (255, 255, 255))
            x, y, w, h = det.bbox

            # 마스크 오버레이
            if show_mask and det.mask is not None:
                # 마스크를 프레임 크기로 리사이즈
                mask_resized = cv2.resize(det.mask, (width, height))
                mask_bool = mask_resized > 0.5

                # 마스크 영역에 색상 오버레이
                overlay = output.copy()
                overlay[mask_bool] = color
                output = cv2.addWeighted(output, 1 - mask_alpha, overlay, mask_alpha, 0)

            # 바운딩 박스
            if show_bbox:
                thickness = 3 if det.is_fallen or det.is_fighting else 2
                cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

            # 레이블
            if show_label:
                label = f"{det.class_name} {det.confidence:.0%}"

                # 배경 박스
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(output,
                             (x, y - label_h - 10),
                             (x + label_w + 10, y),
                             color, -1)

                # 텍스트
                cv2.putText(output, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 프레임 번호
        cv2.putText(output, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 상태 요약
        fallen_count = sum(1 for d in detections if d.is_fallen)
        fighting_count = sum(1 for d in detections if d.is_fighting)

        if fallen_count > 0:
            cv2.putText(output, f"FALLEN: {fallen_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if fighting_count > 0:
            cv2.putText(output, f"FIGHTING: {fighting_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        return output

    def release(self):
        """리소스 해제"""
        pass


# 편의 함수
def create_detector(model_size: str = 'm', **kwargs) -> SegmentationDetector:
    """
    감지기 생성 헬퍼

    Args:
        model_size: 'm' 또는 'l'
        **kwargs: SegmentationDetector 추가 인자
    """
    model_path = f'models/best_{model_size}.pt'
    return SegmentationDetector(model_path=model_path, **kwargs)


# 테스트
if __name__ == "__main__":
    import sys
    import time

    # 감지기 생성
    detector = create_detector('m', confidence=0.3)

    # 비디오 테스트
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'videos/승강기1.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        sys.exit(1)

    print(f"비디오 로드: {video_path}")
    print("ESC 또는 'q'로 종료")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 감지
        detections = detector.process_frame(frame, frame_number=frame_num)

        # 시각화
        output = detector.draw_results(frame, detections)

        # FPS 표시
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (10, output.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 이벤트 출력
        for det in detections:
            if det.is_fallen:
                print(f"[쓰러짐 감지] 프레임 {frame_num}, 신뢰도: {det.confidence:.2f}")
            if det.is_fighting:
                print(f"[싸움 감지] 프레임 {frame_num}, 신뢰도: {det.confidence:.2f}")

        cv2.imshow("Segmentation Detector", output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n처리 완료: {frame_count} 프레임, 평균 {fps:.1f} FPS")
