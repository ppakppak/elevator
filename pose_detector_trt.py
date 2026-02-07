#!/usr/bin/env python3
"""
TensorRT 최적화 YOLO11-Pose 기반 포즈 추정 모듈
쓰러짐 및 싸움 감지 기능 포함
고성능 실시간 처리
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class PersonPose:
    """한 사람의 포즈 정보"""
    landmarks: np.ndarray  # [17, 3] 형태의 키포인트 배열
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]
    is_fallen: bool = False
    fall_confidence: float = 0.0


class TRTPoseDetector:
    """TensorRT 최적화 YOLO11-Pose 포즈 추정 및 이벤트 감지 클래스"""

    # YOLO Pose 키포인트 인덱스 (COCO 포맷, 17개)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # 스켈레톤 연결 정의
    SKELETON = [
        (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    # 색상 정의 (BGR)
    COLORS = {
        'skeleton': (0, 255, 0),      # 초록
        'keypoint': (0, 255, 255),    # 노랑
        'bbox_normal': (0, 255, 0),   # 초록
        'bbox_fallen': (0, 0, 255),   # 빨강
        'text': (255, 255, 255),      # 흰색
    }

    def __init__(self,
                 model_path: str = 'yolo11n-pose.pt',
                 min_detection_confidence: float = 0.5):
        """
        Args:
            model_path: 모델 파일 경로 (.pt, .engine, .onnx)
            min_detection_confidence: 최소 감지 신뢰도
        """
        import os

        # 엔진 파일 우선 시도, 없으면 PT 파일 사용
        engine_path = model_path.replace('.pt', '.engine')
        if model_path.endswith('.pt') and os.path.exists(engine_path):
            model_path = engine_path
            print(f"TensorRT 엔진 발견: {model_path}")

        print(f"포즈 모델 로딩: {model_path}")
        self.model = YOLO(model_path, task='pose')
        self.min_confidence = min_detection_confidence

        # 쓰러짐 감지 임계값
        self.FALL_ANGLE_THRESHOLD = 45
        self.FALL_HEIGHT_RATIO_THRESHOLD = 0.3

        # 싸움 감지 임계값
        self.FIGHT_DISTANCE_THRESHOLD = 150
        self.FIGHT_MOVEMENT_THRESHOLD = 30

        # 이전 프레임의 포즈 정보 저장
        self.previous_poses: List[PersonPose] = []

        print(f"포즈 모델 로드 완료")

    def get_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        """키포인트 가져오기"""
        if keypoints is None or len(keypoints) == 0 or idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        confidence = float(kp[2]) if len(kp) > 2 else 0.0
        if confidence > 0.5:
            return (float(kp[0]), float(kp[1]))
        return None

    def detect_fall(self, keypoints: np.ndarray,
                   frame_width: int,
                   frame_height: int,
                   bbox: Tuple[int, int, int, int] = None) -> Tuple[bool, float]:
        """쓰러짐 감지 (다중 방식 - 천장 카메라 지원)

        1. 바운딩 박스 비율: 가로가 세로보다 길면 쓰러짐
        2. 바운딩 박스 위치: 화면 하단에 있으면 바닥에 있을 가능성
        3. 키포인트 기반: 어깨-엉덩이 수평 여부
        4. 머리-발목 기반: 머리와 발목이 비슷한 높이
        5. 몸통 각도: 어깨-엉덩이 연결선의 기울기
        """
        scores = []

        # 방법 1: 바운딩 박스 비율 (천장 카메라에서 특히 유효)
        # 단, 키포인트로 서있는 자세가 확인되면 이 방법 무시
        is_upright = False
        if bbox is not None:
            x, y, w, h = bbox
            aspect_ratio = w / max(h, 1)

            # 바운딩 박스가 너무 크면 여러 사람일 가능성 -> 무시
            bbox_area_ratio = (w * h) / (frame_width * frame_height)
            if bbox_area_ratio > 0.4:  # 화면의 40% 이상 차지하면 여러 사람
                pass  # bbox_ratio 점수 추가 안함
            # 가로가 세로보다 1.1배 이상 길면 쓰러짐 가능성 (1.2에서 1.1로 낮춤)
            elif aspect_ratio > 1.1:
                # 1.1~2.0 범위에서 0~1로 스케일링
                bbox_score = min((aspect_ratio - 1.0) / 0.9, 1.0)
                scores.append(('bbox_ratio', bbox_score * 0.7))

        if keypoints is None or len(keypoints) < 17:
            if scores:
                total = sum(s[1] for s in scores)
                return total > 0.4, total
            return False, 0.0

        # 방법 2: 어깨-엉덩이 기반 (Y축 높이 차이)
        left_shoulder = self.get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_hip = self.get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self.get_keypoint(keypoints, self.RIGHT_HIP)
        nose = self.get_keypoint(keypoints, self.NOSE)

        # 서있는 자세 확인: 머리가 어깨보다 위에 있고, 어깨가 엉덩이보다 위에 있으면 서있음
        # 쓰러진 자세: 머리가 어깨보다 아래이거나 수평에 가까움
        is_not_upright = False
        if nose and all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            # 이미지에서 Y좌표는 아래로 갈수록 커짐
            # 서있으면: nose_y < shoulder_y < hip_y
            if nose[1] < shoulder_center_y < hip_center_y:
                # 머리-엉덩이 수직 거리가 일정 이상이면 확실히 서있음
                vertical_span = (hip_center_y - nose[1]) / frame_height
                if vertical_span > 0.15:  # 화면의 15% 이상 세로로 퍼져있음
                    is_upright = True
                    # 서있는 경우 bbox_ratio 점수 제거
                    scores = [s for s in scores if s[0] != 'bbox_ratio']
            else:
                # 머리가 어깨보다 아래이면 쓰러진 상태
                is_not_upright = True
                scores.append(('body_orientation', 0.6))

        # 코가 감지되지 않고 어깨-엉덩이만 있을 때: 어깨-엉덩이 Y축 차이로 판단
        elif not nose and all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            vertical_diff = abs(hip_center_y - shoulder_center_y) / frame_height
            # 어깨와 엉덩이가 수평에 가까우면 (차이가 작으면) 쓰러진 상태
            if vertical_diff < 0.12:
                fall_score = 1.0 - (vertical_diff / 0.12)
                scores.append(('shoulder_hip_horizontal', fall_score * 0.6))

        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center_y_norm = (left_shoulder[1] + right_shoulder[1]) / 2 / frame_height
            hip_center_y_norm = (left_hip[1] + right_hip[1]) / 2 / frame_height

            height_diff = abs(shoulder_center_y_norm - hip_center_y_norm)
            if height_diff < 0.1 and not is_upright:
                pose_score = 1.0 - (height_diff / 0.1)
                scores.append(('pose_vertical', pose_score * 0.8))

        # 방법 3: 몸통 각도 (천장 카메라용 - 어깨와 엉덩이의 X축 차이)
        # 서있는 자세가 확인되면 이 방법 무시
        if all([left_shoulder, right_shoulder, left_hip, right_hip]) and not is_upright:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            # 어깨-엉덩이 벡터의 길이
            dx = abs(shoulder_center_x - hip_center_x)
            dy = abs(shoulder_center_y - hip_center_y)

            # 천장에서 볼 때: 서있으면 어깨-엉덩이가 거의 같은 위치
            # 누워있으면 어깨-엉덩이가 다른 위치 (몸이 펼쳐짐)
            body_length = np.sqrt(dx**2 + dy**2)
            body_length_norm = body_length / max(frame_width, frame_height)

            # 몸통이 펼쳐져 있으면 (길이가 길면) 쓰러짐
            # 임계값 상향 (0.15 -> 0.25)
            if body_length_norm > 0.25:
                body_score = min((body_length_norm - 0.15) / 0.25, 1.0)
                scores.append(('body_spread', body_score * 0.65))

        # 방법 4: 머리-발목 거리 (천장 카메라용)
        left_ankle = self.get_keypoint(keypoints, self.LEFT_ANKLE)
        right_ankle = self.get_keypoint(keypoints, self.RIGHT_ANKLE)

        if nose and (left_ankle or right_ankle):
            ankle_x = left_ankle[0] if left_ankle else right_ankle[0]
            ankle_y = left_ankle[1] if left_ankle else right_ankle[1]

            # 머리-발목 거리 (2D)
            dist = np.sqrt((nose[0] - ankle_x)**2 + (nose[1] - ankle_y)**2)
            dist_norm = dist / max(frame_width, frame_height)

            # 거리가 멀면 쓰러짐 (몸이 펼쳐짐)
            if dist_norm > 0.2:
                head_ankle_score = min(dist_norm / 0.4, 1.0)
                scores.append(('head_ankle_dist', head_ankle_score * 0.8))

        # 방법 5: 키포인트 분포 (몸이 화면에 넓게 퍼져있는지)
        # 천장 카메라에서만 유효: 가로세로 비율이 비슷하면서 넓게 퍼져있을 때
        # 주의: 서있는 사람이 확인된 경우 또는 bbox가 너무 큰 경우 이 방법 사용 안함
        bbox_too_large = False
        if bbox is not None:
            x, y, w, h = bbox
            bbox_area_ratio = (w * h) / (frame_width * frame_height)
            if bbox_area_ratio > 0.3:  # 화면의 30% 이상 차지하면
                bbox_too_large = True

        if not is_upright and not bbox_too_large:
            valid_kps = []
            for i in range(17):
                kp = self.get_keypoint(keypoints, i)
                if kp:
                    valid_kps.append(kp)

            if len(valid_kps) >= 5:
                xs = [kp[0] for kp in valid_kps]
                ys = [kp[1] for kp in valid_kps]
                spread_x = (max(xs) - min(xs)) / frame_width
                spread_y = (max(ys) - min(ys)) / frame_height

                # 키포인트가 넓게 퍼져있고, X와 Y 방향으로 고르게 퍼져있을 때만
                # (서있는 사람은 Y 방향으로만 길고, 천장에서 본 누운 사람은 둘 다 비슷)
                total_spread = spread_x + spread_y
                spread_ratio = min(spread_x, spread_y) / max(spread_x, spread_y, 0.001)

                # 비율이 0.5 이상이면 (가로세로가 비슷하게 퍼져있음) 천장에서 본 누운 상태
                # Y가 X보다 많이 크면 서있는 것이므로 제외
                if total_spread > 0.35 and spread_ratio > 0.5 and spread_y < spread_x * 2:
                    spread_score = min(total_spread / 0.7, 1.0) * spread_ratio
                    scores.append(('keypoint_spread', spread_score * 0.7))

        # 종합 점수 계산
        if not scores:
            return False, 0.0

        # 점수 합산 방식 (여러 신호가 있으면 더 높은 점수)
        best_score = max(s[1] for s in scores)
        total_score = sum(s[1] for s in scores)
        # 최고 점수와 합산 점수 중 높은 것 사용
        final_score = max(best_score, min(total_score, 1.0))
        is_fallen = final_score > 0.45

        # 디버그: 감지된 방법들 출력
        if is_fallen:
            methods = [f"{name}:{score:.2f}" for name, score in scores]
            print(f"  [Fall] score={final_score:.2f}, methods={methods}")

        return is_fallen, final_score

    def detect_fighting(self, current_poses: List[PersonPose],
                       frame_width: int,
                       frame_height: int) -> Tuple[bool, float]:
        """싸움 감지"""
        if len(current_poses) < 2:
            return False, 0.0

        fighting_score = 0.0

        # 거리 기반 점수
        for i, pose1 in enumerate(current_poses):
            for pose2 in current_poses[i+1:]:
                c1 = (pose1.bbox[0] + pose1.bbox[2] / 2,
                      pose1.bbox[1] + pose1.bbox[3] / 2)
                c2 = (pose2.bbox[0] + pose2.bbox[2] / 2,
                      pose2.bbox[1] + pose2.bbox[3] / 2)

                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

                if dist < self.FIGHT_DISTANCE_THRESHOLD:
                    distance_score = 1.0 - (dist / self.FIGHT_DISTANCE_THRESHOLD)
                    fighting_score = max(fighting_score, distance_score * 0.5)

        # 쓰러짐 연계 점수
        fallen_count = sum(1 for p in current_poses if p.is_fallen)
        if fallen_count > 0 and len(current_poses) >= 2:
            fighting_score += 0.3

        is_fighting = fighting_score > 0.5
        return is_fighting, fighting_score

    def process_frame(self, frame: np.ndarray) -> Tuple[List[PersonPose], bool, float]:
        """프레임 처리 및 이벤트 감지"""
        height, width = frame.shape[:2]
        current_poses: List[PersonPose] = []

        # TensorRT 추론
        results = self.model(frame, conf=self.min_confidence, verbose=False)

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None and hasattr(result.keypoints, 'data'):
                keypoints_tensor = result.keypoints.data

                if hasattr(keypoints_tensor, 'shape') and len(keypoints_tensor.shape) > 0:
                    if keypoints_tensor.shape[0] > 0:
                        keypoints_data = keypoints_tensor.cpu().numpy()
                        boxes_data = result.boxes.xyxy.cpu().numpy()

                        for person_idx in range(keypoints_data.shape[0]):
                            person_keypoints = keypoints_data[person_idx]
                            box = boxes_data[person_idx]

                            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                            # 중심점 계산
                            confidence_mask = person_keypoints[:, 2] > 0.5
                            valid_keypoints = person_keypoints[confidence_mask]

                            if len(valid_keypoints) > 0:
                                center_x = float(np.mean(valid_keypoints[:, 0])) / width
                                center_y = float(np.mean(valid_keypoints[:, 1])) / height
                                center = (center_x, center_y)
                            else:
                                center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)

                            # 쓰러짐 감지 (bbox 포함)
                            is_fallen, fall_confidence = self.detect_fall(
                                person_keypoints, width, height, bbox
                            )

                            pose = PersonPose(
                                landmarks=person_keypoints,
                                bbox=bbox,
                                center=center,
                                is_fallen=is_fallen,
                                fall_confidence=fall_confidence
                            )
                            current_poses.append(pose)

        # 싸움 감지
        is_fighting, fight_confidence = self.detect_fighting(current_poses, width, height)

        self.previous_poses = current_poses

        return current_poses, is_fighting, fight_confidence

    def draw_pose(self, frame: np.ndarray, poses: List[PersonPose],
                  is_fighting: bool, fight_confidence: float):
        """포즈 및 이벤트 시각화"""
        for person_idx, pose in enumerate(poses):
            if pose.landmarks is not None:
                keypoints = pose.landmarks

                # 키포인트 점 그리기
                for idx, kp in enumerate(keypoints):
                    confidence = float(kp[2]) if len(kp) > 2 else 0.0
                    if confidence > 0.5:
                        x, y = int(float(kp[0])), int(float(kp[1]))
                        cv2.circle(frame, (x, y), 4, self.COLORS['keypoint'], -1)

                # 스켈레톤 연결선 그리기
                for start_idx, end_idx in self.SKELETON:
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        start_kp = keypoints[start_idx]
                        end_kp = keypoints[end_idx]
                        if float(start_kp[2]) > 0.5 and float(end_kp[2]) > 0.5:
                            start_pt = (int(float(start_kp[0])), int(float(start_kp[1])))
                            end_pt = (int(float(end_kp[0])), int(float(end_kp[1])))
                            cv2.line(frame, start_pt, end_pt, self.COLORS['skeleton'], 2)

            # 바운딩 박스 그리기
            x, y, w, h = pose.bbox
            color = self.COLORS['bbox_fallen'] if pose.is_fallen else self.COLORS['bbox_normal']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 쓰러짐 정보 표시
            if pose.is_fallen:
                cv2.putText(frame,
                           f"FALLEN! ({pose.fall_confidence:.2f})",
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           self.COLORS['bbox_fallen'],
                           2)

            # 사람 번호 표시
            cv2.putText(frame,
                       f"Person {person_idx + 1}",
                       (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       self.COLORS['text'],
                       1)

        # 싸움 정보 표시
        if is_fighting:
            cv2.putText(frame,
                       f"FIGHTING DETECTED! ({fight_confidence:.2f})",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (0, 0, 255),
                       3)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def release(self):
        """리소스 해제"""
        pass
