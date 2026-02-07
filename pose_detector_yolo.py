"""
YOLOv11-Pose 기반 포즈 추정 모듈
쓰러짐 및 싸움 감지 기능 포함
다중 인물 네이티브 지원

천장 카메라 모드 지원 (앙상블 낙상 감지)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO

# 천장 카메라 앙상블 감지기 임포트
try:
    from ceiling_fall_detector import CeilingFallDetector
    CEILING_DETECTOR_AVAILABLE = True
except ImportError:
    CEILING_DETECTOR_AVAILABLE = False
    print("Warning: ceiling_fall_detector not found. Ceiling mode disabled.")


@dataclass
class PersonPose:
    """한 사람의 포즈 정보"""
    landmarks: np.ndarray  # [17, 3] 형태의 키포인트 배열
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]
    is_fallen: bool = False
    fall_confidence: float = 0.0


class YOLOPoseDetector:
    """YOLOv11-Pose를 사용한 포즈 추정 및 이벤트 감지 클래스"""
    
    # YOLOv11-Pose 키포인트 인덱스 (COCO 포맷, 17개)
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
    
    def __init__(self,
                 model_size='n',
                 min_detection_confidence=0.5,
                 device='auto',
                 ceiling_mode=False):
        """
        Args:
            model_size: 모델 크기 ('n', 's', 'm', 'l', 'x')
            min_detection_confidence: 최소 감지 신뢰도
            device: 사용할 디바이스 ('auto'=자동, 'cpu', 'cuda', '0' 등)
            ceiling_mode: 천장 카메라 모드 (앙상블 낙상 감지 사용)
        """
        # 천장 카메라 모드 설정
        self.ceiling_mode = ceiling_mode and CEILING_DETECTOR_AVAILABLE
        if ceiling_mode and not CEILING_DETECTOR_AVAILABLE:
            print("Warning: Ceiling mode requested but detector not available. Using standard mode.")

        self.ceiling_detector = None
        if self.ceiling_mode:
            self.ceiling_detector = CeilingFallDetector(fall_confirm_frames=3)
            print("천장 카메라 모드 활성화: 앙상블 낙상 감지 사용")

        # YOLOv11-Pose 모델 로드 (Ultralytics가 자동 다운로드)
        # 주의: 모델 이름은 'yolo11' (v 없음)입니다
        model_name = f'yolo11{model_size}-pose.pt'
        print(f"YOLOv11-Pose 모델 로딩: {model_name}")
        print("처음 실행 시 모델이 자동으로 다운로드됩니다...")

        # Ultralytics는 모델 이름만 지정하면 자동으로 다운로드합니다
        # 모델은 현재 디렉토리 또는 ~/.ultralytics/weights/에 저장됩니다
        self.model = YOLO(model_name)
        print(f"모델 로드 완료: {model_name}")

        # 천장 카메라 모드에서는 감지 신뢰도를 낮춤 (쓰러진 사람 감지를 위해)
        if self.ceiling_mode:
            self.min_confidence = 0.25  # 더 낮게 설정
            self.imgsz = 480  # 성능 최적화 (640 -> 480으로 7배 빨라짐)
            print(f"천장 모드: YOLO 감지 신뢰도 = {self.min_confidence}, imgsz = {self.imgsz}")
        else:
            self.min_confidence = min_detection_confidence
            self.imgsz = 640  # 기본값
        
        # 디바이스 자동 선택
        if device == 'auto':
            # Ultralytics의 select_device를 사용하여 최적 디바이스 선택
            try:
                from ultralytics.utils.torch_utils import select_device
                selected_device = select_device('', verbose=False)
                self.device = str(selected_device) if 'cuda' in str(selected_device) else None
                print(f"디바이스: 자동 선택 -> {selected_device}")
            except:
                # 실패 시 Ultralytics가 자동으로 선택하도록 None 사용
                self.device = None
                print("디바이스: 자동 선택 (Ultralytics 기본)")
        else:
            self.device = device
            print(f"디바이스: {device}")
        
        # 쓰러짐 감지 임계값 (상부 카메라/탑뷰 최적화)
        # 탑뷰에서는 서있는 사람도 세로 길이가 짧아 보이므로 임계값 낮춤
        self.ASPECT_RATIO_THRESHOLD = 1.2  # 종횡비: 서있음 > 1.2, 쓰러짐/앉음 < 0.8
        self.SPREAD_RATIO_THRESHOLD = 0.6  # 분산비: 서있음 < 0.6, 쓰러짐 > 1.0
        self.BODY_LENGTH_THRESHOLD = 0.25  # 신체길이: 서있음 > 0.25, 쓰러짐 < 0.15, 앉음 < 0.12
        self.BBOX_AREA_THRESHOLD = 0.03    # 바운딩박스 면적 기준 (탑뷰에서 더 작음)
        self.BBOX_HEIGHT_RATIO_THRESHOLD = 0.33  # 바운딩박스 높이 / 이미지 높이 임계값

        # 시간 필터링 (연속 프레임 감지)
        self.FALL_CONFIRM_FRAMES = 3  # 쓰러짐 확정에 필요한 연속 프레임 수
        self.fall_frame_counts = {}   # 사람별 연속 쓰러짐 감지 카운트

        # 디버그 모드
        self.debug_mode = False
        self.debug_scores = {}  # 마지막 계산된 점수들 (사람별)

        # 싸움 감지 임계값
        self.FIGHT_DISTANCE_THRESHOLD = 150  # 픽셀 단위
        self.FIGHT_MOVEMENT_THRESHOLD = 30  # 픽셀 단위 (프레임 간 이동)

        # 이전 프레임의 포즈 정보 저장
        self.previous_poses: List[PersonPose] = []

        # 프레임 카운터
        self.frame_count = 0
        
        device_str = "자동" if self.device is None else self.device
        mode_str = "천장 카메라 (앙상블)" if self.ceiling_mode else "일반"
        print(f"YOLOv11-Pose 초기화 완료 (모델: {model_name}, 디바이스: {device_str}, 모드: {mode_str})")

        # 워밍업 (첫 추론이 느리므로 더미 이미지로 미리 실행)
        print("모델 워밍업 중...")
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, conf=0.5, imgsz=self.imgsz, verbose=False)
        print("워밍업 완료")
    
    def get_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        """키포인트 가져오기"""
        if keypoints is None or len(keypoints) == 0 or idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        # NumPy 배열의 confidence 값을 스칼라로 변환
        confidence = float(kp[2]) if len(kp) > 2 else 0.0
        if confidence > 0.5:
            return (float(kp[0]), float(kp[1]))
        return None
    
    def calculate_angle(self, p1: Tuple[float, float], 
                       p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산 (도 단위)"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def detect_fall(self, keypoints: np.ndarray,
                   frame_width: int,
                   frame_height: int,
                   person_id: int = 0,
                   yolo_bbox: Tuple[int, int, int, int] = None) -> Tuple[bool, float, dict]:
        """
        쓰러짐 감지 (상부 카메라/탑뷰 최적화)
        쓰러짐과 쪼그려 앉음 모두 감지

        하체(엉덩이, 무릎, 발목) 감지 여부에 따라 로직 분기:
        - 하체 감지됨: 머리-하체 거리 기반 판단
        - 하체 미감지: 상체만으로 판단 (쪼그림 가능성 높음)

        Args:
            keypoints: [17, 3] 형태의 키포인트 배열 (x, y, confidence)
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            person_id: 사람 ID (시간 필터링용)
            yolo_bbox: YOLO 바운딩 박스 (x, y, width, height)

        Returns:
            (is_fallen, confidence, debug_info): 쓰러짐 여부, 신뢰도, 디버그 정보
        """
        debug_info = {
            'mode': 'none', 'aspect': 0.0, 'spread': 0.0, 'length': 0.0,
            'area': 0.0, 'raw_conf': 0.0, 'upper_compact': 0.0,
            'bbox_height_ratio': 0.0
        }

        if keypoints is None or len(keypoints) < 17:
            return False, 0.0, debug_info

        # === 0. YOLO 바운딩 박스 높이 비율 체크 (최우선) ===
        # YOLO bbox 기준으로 높이 비율 계산
        if yolo_bbox is not None:
            yolo_x, yolo_y, yolo_w, yolo_h = yolo_bbox
            bbox_height_ratio = yolo_h / frame_height
        else:
            # fallback: 키포인트 기준
            valid_mask = keypoints[:, 2] > 0.5
            valid_kps_temp = keypoints[valid_mask]
            if len(valid_kps_temp) >= 3:
                min_y = np.min(valid_kps_temp[:, 1])
                max_y = np.max(valid_kps_temp[:, 1])
                bbox_height_ratio = (max_y - min_y) / frame_height
            else:
                bbox_height_ratio = 1.0  # 키포인트 부족시 정상으로 간주

        # 이미지 비율에 따라 임계값 동적 설정
        # 세로로 긴 이미지: 0.33, 가로로 긴 이미지: 0.5
        if frame_height > frame_width:
            # 세로로 긴 이미지 (portrait)
            height_ratio_threshold = 0.33
        else:
            # 가로로 긴 이미지 (landscape)
            height_ratio_threshold = 0.5

        # 유효한 키포인트 필터링
        valid_mask = keypoints[:, 2] > 0.5
        valid_keypoints = keypoints[valid_mask]

        if len(valid_keypoints) < 3:
            return False, 0.0, debug_info

        # 눈 감지 여부 확인 (눈이 보이면 서있는 상태로 판단)
        left_eye = self.get_keypoint(keypoints, self.LEFT_EYE)
        right_eye = self.get_keypoint(keypoints, self.RIGHT_EYE)
        has_eye = left_eye is not None or right_eye is not None

        # 바운딩 박스 높이가 임계값 이하면 즉시 쓰러짐 판정
        # 단, 눈이 감지되면 쓰러짐으로 처리하지 않음
        if bbox_height_ratio <= height_ratio_threshold and not has_eye:
            raw_confidence = 1.0 - (bbox_height_ratio / height_ratio_threshold)
            raw_confidence = max(0.6, min(1.0, raw_confidence))  # 최소 0.6 보장

            debug_info = {
                'mode': 'bbox_height',
                'bbox_height_ratio': bbox_height_ratio,
                'height_ratio_threshold': height_ratio_threshold,
                'raw_conf': raw_confidence,
                'valid_kp_count': len(valid_keypoints),
                'has_eye': has_eye
            }
            self.debug_scores[person_id] = debug_info

            if self.debug_mode and person_id == 0:
                print(f"[DEBUG-bbox_height] height_ratio={bbox_height_ratio:.3f} (<= {height_ratio_threshold}) "
                      f"-> FALLEN conf={raw_confidence:.3f}")

            # 시간 필터링 적용
            self.fall_frame_counts[person_id] = self.fall_frame_counts.get(person_id, 0) + 1
            is_fallen = self.fall_frame_counts.get(person_id, 0) >= self.FALL_CONFIRM_FRAMES

            if is_fallen:
                return True, raw_confidence, debug_info
            else:
                frame_ratio = self.fall_frame_counts.get(person_id, 0) / self.FALL_CONFIRM_FRAMES
                return False, raw_confidence * frame_ratio, debug_info

        # bbox 조건 충족했지만 눈이 감지된 경우 → fall_frame_count 리셋
        if bbox_height_ratio <= height_ratio_threshold and has_eye:
            self.fall_frame_counts[person_id] = 0
            if self.debug_mode and person_id == 0:
                print(f"[DEBUG-bbox_height] height_ratio={bbox_height_ratio:.3f} but eye detected -> NOT FALLEN")

        # === 키포인트 추출 ===
        # 상체
        nose = self.get_keypoint(keypoints, self.NOSE)
        left_eye = self.get_keypoint(keypoints, self.LEFT_EYE)
        right_eye = self.get_keypoint(keypoints, self.RIGHT_EYE)
        left_shoulder = self.get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(keypoints, self.RIGHT_SHOULDER)

        # 하체
        left_hip = self.get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self.get_keypoint(keypoints, self.RIGHT_HIP)
        left_knee = self.get_keypoint(keypoints, self.LEFT_KNEE)
        right_knee = self.get_keypoint(keypoints, self.RIGHT_KNEE)
        left_ankle = self.get_keypoint(keypoints, self.LEFT_ANKLE)
        right_ankle = self.get_keypoint(keypoints, self.RIGHT_ANKLE)

        # 하체 키포인트 존재 여부 확인
        has_hip = left_hip is not None or right_hip is not None
        has_knee = left_knee is not None or right_knee is not None
        has_ankle = left_ankle is not None or right_ankle is not None
        has_lower_body = has_hip or has_knee or has_ankle

        # 상체 키포인트 존재 여부
        has_head = nose is not None or left_eye is not None or right_eye is not None
        has_shoulder = left_shoulder is not None or right_shoulder is not None

        if not has_head and not has_shoulder:
            return False, 0.0, debug_info

        # 키포인트 바운딩 박스 계산 (valid_keypoints는 이미 위에서 계산됨)
        min_x = np.min(valid_keypoints[:, 0])
        min_y = np.min(valid_keypoints[:, 1])
        max_y = np.max(valid_keypoints[:, 1])
        max_x = np.max(valid_keypoints[:, 0])
        bbox_width = max_x - min_x + 1e-6
        bbox_height = (max_y - min_y) + 1e-6
        aspect_ratio = bbox_height / bbox_width

        # 바운딩 박스 면적 (정규화)
        bbox_area = bbox_width * bbox_height
        normalized_area = bbox_area / (frame_width * frame_height)

        # ========================================
        # 로직 분기: 하체 감지 여부에 따라
        # ========================================

        if has_lower_body:
            # ====== MODE A: 하체가 감지된 경우 ======
            # 관절 각도 기반 판단 (쪼그림: 엉덩이/무릎 각도가 작음)
            mode = 'lower_detected'

            # === 1. 엉덩이 관절 각도 (어깨-엉덩이-무릎) ===
            # 쪼그리면 이 각도가 작아짐 (90도 이하)
            # 서있으면 거의 180도에 가까움
            hip_angle_left = None
            hip_angle_right = None
            hip_angle_score = 0.0

            if left_shoulder and left_hip and left_knee:
                hip_angle_left = self.calculate_angle(left_shoulder, left_hip, left_knee)
            if right_shoulder and right_hip and right_knee:
                hip_angle_right = self.calculate_angle(right_shoulder, right_hip, right_knee)

            # 평균 엉덩이 각도 계산
            hip_angles = [a for a in [hip_angle_left, hip_angle_right] if a is not None]
            avg_hip_angle = np.mean(hip_angles) if hip_angles else None

            if avg_hip_angle is not None:
                # 서있음: 각도 > 150도 → score = 0
                # 쪼그림: 각도 < 120도 → score 증가
                HIP_ANGLE_THRESHOLD = 140  # 이보다 작으면 쪼그림 가능성
                if avg_hip_angle < HIP_ANGLE_THRESHOLD:
                    hip_angle_score = 1.0 - (avg_hip_angle / HIP_ANGLE_THRESHOLD)
                hip_angle_score = max(0.0, min(1.0, hip_angle_score))

            # === 2. 무릎 관절 각도 (엉덩이-무릎-발목) ===
            # 쪼그리면 무릎이 구부러져 각도가 작아짐 (90도 이하)
            # 서있으면 거의 180도에 가까움
            knee_angle_left = None
            knee_angle_right = None
            knee_angle_score = 0.0

            if left_hip and left_knee and left_ankle:
                knee_angle_left = self.calculate_angle(left_hip, left_knee, left_ankle)
            if right_hip and right_knee and right_ankle:
                knee_angle_right = self.calculate_angle(right_hip, right_knee, right_ankle)

            # 평균 무릎 각도 계산
            knee_angles = [a for a in [knee_angle_left, knee_angle_right] if a is not None]
            avg_knee_angle = np.mean(knee_angles) if knee_angles else None

            if avg_knee_angle is not None:
                # 서있음: 각도 > 160도 → score = 0
                # 쪼그림: 각도 < 120도 → score 증가
                KNEE_ANGLE_THRESHOLD = 140  # 이보다 작으면 쪼그림 가능성
                if avg_knee_angle < KNEE_ANGLE_THRESHOLD:
                    knee_angle_score = 1.0 - (avg_knee_angle / KNEE_ANGLE_THRESHOLD)
                knee_angle_score = max(0.0, min(1.0, knee_angle_score))

            # === 3. 머리-하체 거리 (보조 지표) ===
            head_pos = nose
            if head_pos is None:
                if left_eye and right_eye:
                    head_pos = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
                elif left_eye:
                    head_pos = left_eye
                elif right_eye:
                    head_pos = right_eye
                elif left_shoulder and right_shoulder:
                    head_pos = ((left_shoulder[0] + right_shoulder[0]) / 2,
                               (left_shoulder[1] + right_shoulder[1]) / 2)

            lower_pos = None
            if left_ankle and right_ankle:
                lower_pos = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
            elif left_ankle:
                lower_pos = left_ankle
            elif right_ankle:
                lower_pos = right_ankle
            elif left_knee and right_knee:
                lower_pos = ((left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2)
            elif left_knee:
                lower_pos = left_knee
            elif right_knee:
                lower_pos = right_knee
            elif left_hip and right_hip:
                lower_pos = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            elif left_hip:
                lower_pos = left_hip
            elif right_hip:
                lower_pos = right_hip

            body_length_normalized = 0.0
            length_score = 0.0
            if head_pos and lower_pos:
                body_length = np.sqrt((head_pos[0] - lower_pos[0])**2 + (head_pos[1] - lower_pos[1])**2)
                body_length_normalized = body_length / max(frame_width, frame_height)

                if body_length_normalized < self.BODY_LENGTH_THRESHOLD:
                    length_score = 1.0 - (body_length_normalized / self.BODY_LENGTH_THRESHOLD)
                length_score = max(0.0, min(1.0, length_score))

            # === 4. 종횡비 (쓰러짐 감지용) ===
            aspect_score = 0.0
            if aspect_ratio < self.ASPECT_RATIO_THRESHOLD:
                aspect_score = 1.0 - (aspect_ratio / self.ASPECT_RATIO_THRESHOLD)
            aspect_score = max(0.0, min(1.0, aspect_score))

            # === 종합 신뢰도 계산 ===
            # 관절 각도가 있으면 각도 기반, 없으면 거리/종횡비 기반
            has_angle_data = avg_hip_angle is not None or avg_knee_angle is not None

            if has_angle_data:
                # 관절 각도 기반 (쪼그림 감지에 최적)
                # 둘 다 있으면 평균, 하나만 있으면 그 값 사용
                if avg_hip_angle is not None and avg_knee_angle is not None:
                    raw_confidence = (
                        hip_angle_score * 0.35 +     # 엉덩이 각도
                        knee_angle_score * 0.35 +    # 무릎 각도
                        length_score * 0.20 +        # 머리-하체 거리
                        aspect_score * 0.10          # 종횡비
                    )
                elif avg_hip_angle is not None:
                    raw_confidence = (
                        hip_angle_score * 0.50 +
                        length_score * 0.30 +
                        aspect_score * 0.20
                    )
                else:  # avg_knee_angle is not None
                    raw_confidence = (
                        knee_angle_score * 0.50 +
                        length_score * 0.30 +
                        aspect_score * 0.20
                    )
            else:
                # 관절 각도 없음 → 거리/종횡비 기반 (쓰러짐 감지)
                x_std = np.std(valid_keypoints[:, 0])
                y_std = np.std(valid_keypoints[:, 1]) + 1e-6
                spread_ratio = x_std / y_std
                spread_score = 0.0
                if spread_ratio > self.SPREAD_RATIO_THRESHOLD:
                    spread_score = min((spread_ratio - self.SPREAD_RATIO_THRESHOLD) / 0.5, 1.0)

                raw_confidence = (
                    length_score * 0.40 +
                    aspect_score * 0.35 +
                    spread_score * 0.25
                )
                avg_hip_angle = None
                avg_knee_angle = None

            debug_info = {
                'mode': mode,
                'hip_angle': avg_hip_angle,
                'knee_angle': avg_knee_angle,
                'hip_angle_score': hip_angle_score,
                'knee_angle_score': knee_angle_score,
                'aspect': aspect_score,
                'length': length_score,
                'raw_conf': raw_confidence,
                'aspect_ratio': aspect_ratio,
                'body_length': body_length_normalized,
                'normalized_area': normalized_area,
                'valid_kp_count': len(valid_keypoints),
                'has_hip': has_hip,
                'has_knee': has_knee,
                'has_ankle': has_ankle,
                'has_angle_data': has_angle_data,
                'bbox_height_ratio': bbox_height_ratio
            }

        else:
            # ====== MODE B: 하체가 감지되지 않은 경우 ======
            # 상체만 보임 → 쪼그려 앉음 가능성 높음
            mode = 'upper_only'

            # 상체 키포인트들의 컴팩트함 측정
            # 쪼그리면 상체 키포인트들이 좁은 영역에 모여있음

            # 상체 키포인트만 추출
            upper_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE,
                            self.LEFT_EAR, self.RIGHT_EAR,
                            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                            self.LEFT_ELBOW, self.RIGHT_ELBOW,
                            self.LEFT_WRIST, self.RIGHT_WRIST]
            upper_keypoints = []
            for idx in upper_indices:
                kp = self.get_keypoint(keypoints, idx)
                if kp is not None:
                    upper_keypoints.append(kp)

            upper_compact_score = 0.0
            upper_bbox_size = 0.0

            if len(upper_keypoints) >= 3:
                upper_kp_array = np.array(upper_keypoints)
                upper_min_x = np.min(upper_kp_array[:, 0])
                upper_max_x = np.max(upper_kp_array[:, 0])
                upper_min_y = np.min(upper_kp_array[:, 1])
                upper_max_y = np.max(upper_kp_array[:, 1])

                upper_width = upper_max_x - upper_min_x + 1e-6
                upper_height = upper_max_y - upper_min_y + 1e-6
                upper_bbox_size = np.sqrt(upper_width**2 + upper_height**2) / max(frame_width, frame_height)

                # 상체가 작은 영역에 모여있으면 쪼그림
                # 임계값: 0.15 (이보다 작으면 쪼그림 가능성)
                UPPER_COMPACT_THRESHOLD = 0.15
                if upper_bbox_size < UPPER_COMPACT_THRESHOLD:
                    upper_compact_score = 1.0 - (upper_bbox_size / UPPER_COMPACT_THRESHOLD)
                upper_compact_score = max(0.0, min(1.0, upper_compact_score))

            # 전체 바운딩 박스 면적도 확인
            area_score = 0.0
            if normalized_area < self.BBOX_AREA_THRESHOLD:
                area_score = 1.0 - (normalized_area / self.BBOX_AREA_THRESHOLD)
            area_score = max(0.0, min(1.0, area_score))

            # 종횡비 (상체만 있어도 참고)
            aspect_score = 0.0
            if aspect_ratio < self.ASPECT_RATIO_THRESHOLD:
                aspect_score = 1.0 - (aspect_ratio / self.ASPECT_RATIO_THRESHOLD)
            aspect_score = max(0.0, min(1.0, aspect_score))

            # 하체 미감지 자체가 쪼그림의 강력한 신호
            # 단, 프레임 가장자리에 있어서 잘린 경우는 제외해야 함
            # → 상체가 프레임 중앙 근처에 있는지 확인
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            margin = 0.1  # 프레임 가장자리 10%
            is_near_edge = (center_x < frame_width * margin or
                          center_x > frame_width * (1 - margin) or
                          center_y < frame_height * margin or
                          center_y > frame_height * (1 - margin))

            edge_penalty = 0.5 if is_near_edge else 0.0

            # 종합 신뢰도 (상체만 감지 모드)
            # 하체가 안 보이면 쪼그림 가능성 높음
            raw_confidence = (
                upper_compact_score * 0.40 +  # 상체 컴팩트함이 핵심
                0.30 * (1.0 - edge_penalty) + # 하체 미감지 보너스 (가장자리 아니면)
                area_score * 0.15 +           # 면적
                aspect_score * 0.15           # 종횡비
            )

            debug_info = {
                'mode': mode,
                'aspect': aspect_score,
                'spread': 0.0,
                'length': 0.0,
                'area': area_score,
                'raw_conf': raw_confidence,
                'aspect_ratio': aspect_ratio,
                'spread_ratio': 0.0,
                'body_length': 0.0,
                'normalized_area': normalized_area,
                'valid_kp_count': len(valid_keypoints),
                'upper_compact': upper_compact_score,
                'upper_bbox_size': upper_bbox_size,
                'is_near_edge': is_near_edge,
                'has_hip': has_hip,
                'has_knee': has_knee,
                'has_ankle': has_ankle,
                'bbox_height_ratio': bbox_height_ratio
            }

        self.debug_scores[person_id] = debug_info

        # 디버그 모드: 콘솔에 실제 값 출력
        if self.debug_mode and person_id == 0:
            if mode == 'lower_detected':
                hip_str = f"{avg_hip_angle:.1f}" if avg_hip_angle else "N/A"
                knee_str = f"{avg_knee_angle:.1f}" if avg_knee_angle else "N/A"
                print(f"[DEBUG-{mode}] hip_angle={hip_str}, knee_angle={knee_str}, "
                      f"body_len={body_length_normalized:.3f}, "
                      f"scores=[Hip:{hip_angle_score:.2f} Knee:{knee_angle_score:.2f} Len:{length_score:.2f}] "
                      f"conf={raw_confidence:.3f}")
            else:
                print(f"[DEBUG-{mode}] upper_size={upper_bbox_size:.3f}, compact={upper_compact_score:.2f}, "
                      f"area={normalized_area:.4f}, edge={is_near_edge}, conf={raw_confidence:.3f}")

        # === 시간 필터링 (3프레임 연속 감지) ===
        is_fallen_raw = raw_confidence > 0.5

        if is_fallen_raw:
            self.fall_frame_counts[person_id] = self.fall_frame_counts.get(person_id, 0) + 1
        else:
            self.fall_frame_counts[person_id] = 0

        # 연속 N프레임 이상 감지되어야 최종 쓰러짐 판정
        is_fallen = self.fall_frame_counts.get(person_id, 0) >= self.FALL_CONFIRM_FRAMES

        # 최종 신뢰도 (시간 필터링 반영)
        if is_fallen:
            confidence = raw_confidence
        else:
            # 아직 확정되지 않은 경우 프레임 카운트 비율 반영
            frame_ratio = self.fall_frame_counts.get(person_id, 0) / self.FALL_CONFIRM_FRAMES
            confidence = raw_confidence * frame_ratio

        return is_fallen, confidence, debug_info
    
    def calculate_distance(self, p1: Tuple[float, float], 
                          p2: Tuple[float, float], 
                          frame_width: int, 
                          frame_height: int) -> float:
        """두 점 사이의 거리 계산 (픽셀 단위)"""
        x1, y1 = p1[0] * frame_width, p1[1] * frame_height
        x2, y2 = p2[0] * frame_width, p2[1] * frame_height
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def is_point_in_bbox(self, point: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> bool:
        """점이 바운딩 박스 안에 있는지 확인"""
        if point is None:
            return False
        px, py = point
        bx, by, bw, bh = bbox
        return bx <= px <= bx + bw and by <= py <= by + bh

    def is_bbox_overlapping(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
        """두 바운딩 박스가 겹치는지 확인"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # bbox1의 범위
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1

        # bbox2의 범위
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2

        # 겹침 확인: 하나라도 겹치지 않으면 False
        if right1 < left2 or right2 < left1:
            return False
        if bottom1 < top2 or bottom2 < top1:
            return False

        return True

    def detect_fighting(self, current_poses: List[PersonPose],
                       frame_width: int,
                       frame_height: int) -> Tuple[bool, float]:
        """
        싸움 감지

        Args:
            current_poses: 현재 프레임의 포즈 리스트
            frame_width: 프레임 너비
            frame_height: 프레임 높이

        Returns:
            (is_fighting, confidence): 싸움 여부와 신뢰도
        """
        if len(current_poses) < 2:
            return False, 0.0

        fighting_score = 0.0
        hand_intrusion_detected = False

        # 1. 손 관절이 상대방 얼굴에 더 가까운지 확인 (핵심 지표)
        # 조건: 두 사람 모두 얼굴(눈/코/귀)이 감지되어야 함
        # 한 사람의 손이 자신의 얼굴보다 상대방 얼굴에 더 가까우면 싸움

        def get_face_center(landmarks):
            """얼굴 중심점 계산 (눈, 코 중 감지된 것들의 평균)"""
            face_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE]
            face_points = []
            for idx in face_indices:
                pt = self.get_keypoint(landmarks, idx)
                if pt is not None:
                    face_points.append(pt)
            if len(face_points) == 0:
                return None
            avg_x = np.mean([p[0] for p in face_points])
            avg_y = np.mean([p[1] for p in face_points])
            return (avg_x, avg_y)

        def has_face(landmarks):
            """얼굴(눈/코) 중 하나라도 감지되었는지 확인"""
            face_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE]
            for idx in face_indices:
                if self.get_keypoint(landmarks, idx) is not None:
                    return True
            return False

        def distance(p1, p2):
            """두 점 사이의 거리"""
            if p1 is None or p2 is None:
                return float('inf')
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        for i, pose_a in enumerate(current_poses):
            if pose_a.landmarks is None:
                continue
            if not has_face(pose_a.landmarks):
                continue

            face_a = get_face_center(pose_a.landmarks)
            if face_a is None:
                continue

            # 손목 키포인트 추출
            left_wrist = self.get_keypoint(pose_a.landmarks, self.LEFT_WRIST)
            right_wrist = self.get_keypoint(pose_a.landmarks, self.RIGHT_WRIST)
            wrists = [w for w in [left_wrist, right_wrist] if w is not None]

            if not wrists:
                continue

            for j, pose_b in enumerate(current_poses):
                if i == j:
                    continue
                if pose_b.landmarks is None:
                    continue
                if not has_face(pose_b.landmarks):
                    continue

                face_b = get_face_center(pose_b.landmarks)
                if face_b is None:
                    continue

                # 각 손목에 대해 검사
                for wrist in wrists:
                    dist_to_own_face = distance(wrist, face_a)
                    dist_to_other_face = distance(wrist, face_b)

                    # 손이 자신의 얼굴보다 상대방 얼굴에 더 가까우면 싸움
                    if dist_to_other_face < dist_to_own_face:
                        hand_intrusion_detected = True
                        fighting_score += 0.7
                        if self.debug_mode:
                            print(f"[DEBUG-fight] Person {i}'s hand closer to Person {j}'s face! "
                                  f"(own:{dist_to_own_face:.1f} vs other:{dist_to_other_face:.1f})")
                        break

                if hand_intrusion_detected:
                    break

            if hand_intrusion_detected:
                break

        # 2. 사람들 간의 거리 확인 (보조 지표)
        if not hand_intrusion_detected:
            distance_scores = []
            for i in range(len(current_poses)):
                for j in range(i + 1, len(current_poses)):
                    dist = self.calculate_distance(
                        current_poses[i].center,
                        current_poses[j].center,
                        frame_width,
                        frame_height
                    )
                    # 거리가 가까울수록 싸움 가능성 증가
                    if dist < self.FIGHT_DISTANCE_THRESHOLD:
                        distance_score = 1.0 - (dist / self.FIGHT_DISTANCE_THRESHOLD)
                        distance_scores.append(distance_score)

            if distance_scores:
                fighting_score += max(distance_scores) * 0.3

        # 3. 움직임 패턴 확인 (이전 프레임과 비교)
        movement_scores = []
        if self.previous_poses:
            for curr_pose in current_poses:
                # 가장 가까운 이전 포즈 찾기
                min_dist = float('inf')
                closest_prev_pose = None

                for prev_pose in self.previous_poses:
                    dist = self.calculate_distance(
                        curr_pose.center,
                        prev_pose.center,
                        frame_width,
                        frame_height
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_prev_pose = prev_pose

                # 빠른 움직임 감지
                if closest_prev_pose and min_dist < self.FIGHT_DISTANCE_THRESHOLD:
                    if min_dist > self.FIGHT_MOVEMENT_THRESHOLD:
                        movement_score = min(min_dist / self.FIGHT_DISTANCE_THRESHOLD, 1.0)
                        movement_scores.append(movement_score)

        if movement_scores:
            fighting_score += max(movement_scores) * 0.2

        # 4. 쓰러짐 감지와 연계 (싸움 중 쓰러짐)
        fall_scores = [pose.fall_confidence for pose in current_poses if pose.is_fallen]
        if fall_scores and len(current_poses) >= 2:
            fighting_score += max(fall_scores) * 0.1

        # 최종 판정
        fighting_score = min(1.0, fighting_score)
        is_fighting = fighting_score > 0.5
        return is_fighting, fighting_score
    
    def process_frame(self, frame: np.ndarray, frame_number: int = None) -> Tuple[List[PersonPose], bool, float]:
        """
        프레임 처리 및 이벤트 감지

        Args:
            frame: 입력 프레임 (BGR)
            frame_number: 영상의 실제 프레임 번호 (옵션)

        Returns:
            (poses, is_fighting, fight_confidence): 포즈 리스트, 싸움 여부, 싸움 신뢰도
        """
        height, width = frame.shape[:2]
        current_poses: List[PersonPose] = []

        # 프레임 번호 설정 (외부에서 전달받거나 내부 카운터 사용)
        if frame_number is not None:
            self.frame_count = frame_number
        else:
            self.frame_count += 1

        # YOLOv11-Pose로 포즈 추정 (다중 인물 자동 감지)
        # device를 None으로 설정하면 Ultralytics가 자동으로 최적 디바이스 선택
        # Jetson의 경우 자동으로 GPU를 사용하려고 시도
        # 'cpu'가 명시적으로 지정되지 않았으면 자동 선택
        inference_device = self.device if self.device == 'cpu' else None
        results = self.model(frame, conf=self.min_confidence, imgsz=self.imgsz,
                            device=inference_device, verbose=False)
        
        # results가 리스트인지 확인하고 안전하게 처리
        try:
            # results는 Results 객체이므로 리스트처럼 동작하지만 안전하게 체크
            if results is None:
                return current_poses, False, 0.0
            
            # Results 객체는 리스트처럼 동작하지만, 직접 len()을 호출하는 대신
            # 리스트로 변환하거나 hasattr로 확인
            results_list = list(results) if hasattr(results, '__iter__') else [results]
            
            if len(results_list) > 0:
                result = results_list[0]
                
                # 키포인트와 바운딩 박스 가져오기
                if (result.keypoints is not None and 
                    hasattr(result.keypoints, 'data')):
                    keypoints_tensor = result.keypoints.data
                    # 텐서의 크기를 안전하게 확인
                    if hasattr(keypoints_tensor, 'shape') and len(keypoints_tensor.shape) > 0:
                        if keypoints_tensor.shape[0] > 0:
                            keypoints_data = keypoints_tensor.cpu().numpy()  # [num_people, 17, 3]
                            boxes_data = result.boxes.xyxy.cpu().numpy()  # [num_people, 4] (x1, y1, x2, y2)
                            
                            num_people = keypoints_data.shape[0] if len(keypoints_data.shape) > 0 else 0
                            for person_idx in range(num_people):
                                person_keypoints = keypoints_data[person_idx]  # [17, 3]
                                box = boxes_data[person_idx]  # [x1, y1, x2, y2]
                                
                                # 바운딩 박스 변환 (배열을 스칼라로 변환)
                                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            
                                # 중심점 계산 (키포인트 기반)
                                # NumPy 배열의 boolean 인덱싱을 안전하게 처리
                                try:
                                    confidence_mask = person_keypoints[:, 2] > 0.5
                                    if isinstance(confidence_mask, np.ndarray):
                                        valid_keypoints = person_keypoints[confidence_mask]
                                    else:
                                        valid_keypoints = person_keypoints
                                    
                                    if len(valid_keypoints) > 0:
                                        center_x = float(np.mean(valid_keypoints[:, 0])) / width
                                        center_y = float(np.mean(valid_keypoints[:, 1])) / height
                                        center = (center_x, center_y)
                                    else:
                                        # 키포인트가 없으면 바운딩 박스 중심 사용
                                        center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
                                except Exception as e:
                                    # 오류 발생 시 바운딩 박스 중심 사용
                                    center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
                                
                                # 쓰러짐 감지 (person_id로 시간 필터링, YOLO bbox 전달)
                                try:
                                    if self.ceiling_mode and self.ceiling_detector is not None:
                                        # 천장 카메라 모드: 앙상블 감지기 사용
                                        result = self.ceiling_detector.detect(
                                            frame, person_idx, bbox, person_keypoints
                                        )
                                        is_fallen = result.is_fallen
                                        fall_confidence = result.confidence
                                        # 디버그 정보 저장
                                        self.debug_scores[person_idx] = {
                                            'mode': 'ceiling_ensemble',
                                            'bbox_score': result.bbox_score,
                                            'motion_score': result.motion_score,
                                            'head_score': result.head_score,
                                            'keypoint_score': result.keypoint_score,
                                            'raw_conf': fall_confidence,
                                            **result.debug_info
                                        }
                                    else:
                                        # 일반 모드: 기존 감지 방식
                                        is_fallen, fall_confidence, debug_info = self.detect_fall(
                                            person_keypoints, width, height,
                                            person_id=person_idx, yolo_bbox=bbox
                                        )
                                except Exception as e:
                                    print(f"쓰러짐 감지 오류: {e}")
                                    is_fallen, fall_confidence = False, 0.0
                                
                                pose = PersonPose(
                                    landmarks=person_keypoints,
                                    bbox=bbox,
                                    center=center,
                                    is_fallen=is_fallen,
                                    fall_confidence=fall_confidence
                                )
                                current_poses.append(pose)
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 싸움 감지
        is_fighting, fight_confidence = self.detect_fighting(current_poses, width, height)

        # 쓰러짐과 싸움이 동시 발생일 경우 쓰러짐만 인정
        has_fallen = any(pose.is_fallen for pose in current_poses)
        if has_fallen and is_fighting:
            is_fighting = False
            fight_confidence = 0.0
            if self.debug_mode:
                print("[DEBUG] Fall and fight detected simultaneously -> only fall recognized")

        # 이전 포즈 업데이트
        self.previous_poses = current_poses

        return current_poses, is_fighting, fight_confidence
    
    def draw_pose(self, frame: np.ndarray, poses: List[PersonPose],
                  is_fighting: bool, fight_confidence: float):
        """포즈 및 이벤트 시각화"""
        height, width = frame.shape[:2]

        # 프레임 번호 표시 (좌측 상단)
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # 검정 테두리
        cv2.putText(frame, frame_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 흰색 텍스트

        # YOLOv11-Pose 키포인트 연결 (COCO 포맷)
        POSE_CONNECTIONS = [
            (self.NOSE, self.LEFT_EYE),
            (self.NOSE, self.RIGHT_EYE),
            (self.LEFT_EYE, self.LEFT_EAR),
            (self.RIGHT_EYE, self.RIGHT_EAR),
            (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            (self.LEFT_SHOULDER, self.LEFT_ELBOW),
            (self.LEFT_ELBOW, self.LEFT_WRIST),
            (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
            (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            (self.LEFT_SHOULDER, self.LEFT_HIP),
            (self.RIGHT_SHOULDER, self.RIGHT_HIP),
            (self.LEFT_HIP, self.RIGHT_HIP),
            (self.LEFT_HIP, self.LEFT_KNEE),
            (self.LEFT_KNEE, self.LEFT_ANKLE),
            (self.RIGHT_HIP, self.RIGHT_KNEE),
            (self.RIGHT_KNEE, self.RIGHT_ANKLE),
        ]
        
        for person_idx, pose in enumerate(poses):
            # 키포인트 그리기
            if pose.landmarks is not None:
                keypoints = pose.landmarks
                height, width = frame.shape[:2]
                
                # 키포인트 점 그리기
                for idx, kp in enumerate(keypoints):
                    confidence = float(kp[2]) if len(kp) > 2 else 0.0
                    if confidence > 0.5:  # confidence > 0.5
                        x, y = int(float(kp[0])), int(float(kp[1]))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # 키포인트 연결선 그리기
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                        start_kp = keypoints[start_idx]
                        end_kp = keypoints[end_idx]
                        start_conf = float(start_kp[2]) if len(start_kp) > 2 else 0.0
                        end_conf = float(end_kp[2]) if len(end_kp) > 2 else 0.0
                        if start_conf > 0.5 and end_conf > 0.5:
                            start_pt = (int(float(start_kp[0])), int(float(start_kp[1])))
                            end_pt = (int(float(end_kp[0])), int(float(end_kp[1])))
                            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            
            # 바운딩 박스 그리기
            x, y, w, h = pose.bbox
            color = (0, 0, 255) if pose.is_fallen else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 쓰러짐 정보 표시
            if pose.is_fallen:
                cv2.putText(frame, 
                           f"FALLEN! ({pose.fall_confidence:.2f})",
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 0, 255),
                           2)
            
            # 사람 번호 표시 (enumerate를 사용하여 인덱스 직접 사용)
            cv2.putText(frame,
                       f"Person {person_idx + 1}",
                       (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),
                       2)

            # 디버그 모드: 각 점수 표시
            if self.debug_mode and person_idx in self.debug_scores:
                debug_info = self.debug_scores[person_idx]
                debug_y = y + h + 40
                line_height = 16
                font_scale = 0.4

                mode = debug_info.get('mode', 'none')
                raw_conf = debug_info.get('raw_conf', 0)
                conf_color = (0, 255, 0) if raw_conf < 0.3 else (0, 255, 255) if raw_conf < 0.5 else (0, 0, 255)

                if mode == 'ceiling_ensemble':
                    # 천장 카메라 앙상블 모드
                    # 배경 박스 (파란색 테두리로 구분)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 260, debug_y + line_height * 8 + 5),
                                 (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 260, debug_y + line_height * 8 + 5),
                                 (255, 150, 0), 2)  # 파란색 테두리

                    # 모드 표시
                    cv2.putText(frame, f"MODE: CEILING ENSEMBLE",
                               (x + 5, debug_y + line_height * 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 150, 0), 1)

                    # 개별 감지기 점수
                    bbox_score = debug_info.get('bbox_score', 0)
                    motion_score = debug_info.get('motion_score', 0)
                    head_score = debug_info.get('head_score', 0)
                    kp_score = debug_info.get('keypoint_score', 0)

                    # 점수 색상 (높을수록 빨간색)
                    def score_color(s):
                        if s < 0.3: return (100, 255, 100)
                        elif s < 0.5: return (100, 255, 255)
                        else: return (100, 100, 255)

                    cv2.putText(frame, f"BBox:     {bbox_score:.3f}",
                               (x + 5, debug_y + line_height * 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, score_color(bbox_score), 1)
                    cv2.putText(frame, f"Motion:   {motion_score:.3f}",
                               (x + 5, debug_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, score_color(motion_score), 1)
                    cv2.putText(frame, f"Head:     {head_score:.3f}",
                               (x + 5, debug_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, score_color(head_score), 1)
                    cv2.putText(frame, f"KeyPoint: {kp_score:.3f}",
                               (x + 5, debug_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, score_color(kp_score), 1)

                    cv2.putText(frame, f"------------------------",
                               (x + 5, debug_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)

                    # 가중치 정보 (천장 감지기에서 가져오기)
                    weights_info = debug_info.get('weights', {})
                    cv2.putText(frame, f"Weights: B:{weights_info.get('bbox', 0):.2f} M:{weights_info.get('motion', 0):.2f} H:{weights_info.get('head', 0):.2f}",
                               (x + 5, debug_y + line_height * 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 최종 신뢰도
                    cv2.putText(frame, f"CONF:     {raw_conf:.3f}",
                               (x + 5, debug_y + line_height * 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, conf_color, 1)

                elif mode == 'lower_detected':
                    # 하체 감지 모드 - 관절 각도 기반
                    # 배경 박스
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 250, debug_y + line_height * 8 + 5),
                                 (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 250, debug_y + line_height * 8 + 5),
                                 (100, 100, 100), 1)

                    # 모드 표시
                    has_angle = debug_info.get('has_angle_data', False)
                    mode_text = "LOWER (angle)" if has_angle else "LOWER (dist)"
                    cv2.putText(frame, f"MODE: {mode_text}",
                               (x + 5, debug_y + line_height * 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), 1)

                    # 하체 감지 상태
                    has_hip = debug_info.get('has_hip', False)
                    has_knee = debug_info.get('has_knee', False)
                    has_ankle = debug_info.get('has_ankle', False)
                    lower_status = f"Hip:{int(has_hip)} Knee:{int(has_knee)} Ankle:{int(has_ankle)}"
                    cv2.putText(frame, lower_status,
                               (x + 5, debug_y + line_height * 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 관절 각도 (핵심 지표)
                    hip_angle = debug_info.get('hip_angle')
                    knee_angle = debug_info.get('knee_angle')
                    hip_score = debug_info.get('hip_angle_score', 0)
                    knee_score = debug_info.get('knee_angle_score', 0)

                    hip_str = f"{hip_angle:.1f}deg" if hip_angle else "N/A"
                    knee_str = f"{knee_angle:.1f}deg" if knee_angle else "N/A"

                    # 각도가 작으면 (쪼그림) 노란색으로 강조
                    hip_color = (0, 255, 255) if hip_angle and hip_angle < 140 else (200, 200, 200)
                    knee_color = (0, 255, 255) if knee_angle and knee_angle < 140 else (200, 200, 200)

                    cv2.putText(frame, f"HipAngle:  {hip_str} -> {hip_score:.2f}",
                               (x + 5, debug_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, hip_color, 1)
                    cv2.putText(frame, f"KneeAngle: {knee_str} -> {knee_score:.2f}",
                               (x + 5, debug_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, knee_color, 1)

                    # 보조 지표
                    body_len = debug_info.get('body_length', 0)
                    aspect_ratio = debug_info.get('aspect_ratio', 0)
                    cv2.putText(frame, f"BodyLen:   {body_len:.3f} -> {debug_info.get('length', 0):.2f}",
                               (x + 5, debug_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
                    cv2.putText(frame, f"AspectR:   {aspect_ratio:.2f} -> {debug_info.get('aspect', 0):.2f}",
                               (x + 5, debug_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
                    cv2.putText(frame, f"ValidKP:   {debug_info.get('valid_kp_count', 0)}",
                               (x + 5, debug_y + line_height * 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 최종 신뢰도
                    cv2.putText(frame, f"CONF:      {raw_conf:.3f}",
                               (x + 5, debug_y + line_height * 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, conf_color, 1)

                elif mode == 'upper_only':
                    # 상체만 감지 모드 (쪼그림 가능성)
                    # 배경 박스 (주황색 테두리로 강조)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 230, debug_y + line_height * 6 + 5),
                                 (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 230, debug_y + line_height * 6 + 5),
                                 (0, 165, 255), 1)  # 주황색 테두리

                    # 모드 표시 (주황색으로 강조)
                    cv2.putText(frame, f"MODE: UPPER_ONLY (squat?)",
                               (x + 5, debug_y + line_height * 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)

                    # 상체 컴팩트 지표
                    upper_size = debug_info.get('upper_bbox_size', 0)
                    upper_compact = debug_info.get('upper_compact', 0)
                    cv2.putText(frame, f"UpperSize: {upper_size:.3f} -> compact:{upper_compact:.2f}",
                               (x + 5, debug_y + line_height * 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 가장자리 여부
                    is_edge = debug_info.get('is_near_edge', False)
                    edge_text = "YES (penalty)" if is_edge else "NO (bonus +0.3)"
                    edge_color = (100, 100, 255) if is_edge else (100, 255, 100)
                    cv2.putText(frame, f"NearEdge: {edge_text}",
                               (x + 5, debug_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, edge_color, 1)

                    # 기타 지표
                    cv2.putText(frame, f"Area:    {debug_info.get('normalized_area', 0):.4f} -> {debug_info.get('area', 0):.2f}",
                               (x + 5, debug_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
                    cv2.putText(frame, f"AspectR: {debug_info.get('aspect_ratio', 0):.2f} -> {debug_info.get('aspect', 0):.2f}",
                               (x + 5, debug_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 최종 신뢰도
                    cv2.putText(frame, f"CONF:    {raw_conf:.3f}",
                               (x + 5, debug_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, conf_color, 1)

                elif mode == 'bbox_height':
                    # 바운딩 박스 높이 비율 모드 (즉시 쓰러짐 판정)
                    # 배경 박스 (빨간색 테두리로 강조)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 250, debug_y + line_height * 4 + 5),
                                 (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 250, debug_y + line_height * 4 + 5),
                                 (0, 0, 255), 2)  # 빨간색 테두리

                    # 모드 표시 (빨간색)
                    cv2.putText(frame, f"MODE: BBOX_HEIGHT (FALLEN!)",
                               (x + 5, debug_y + line_height * 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

                    # 높이 비율 (동적 임계값 표시)
                    height_ratio = debug_info.get('bbox_height_ratio', 0)
                    threshold = debug_info.get('height_ratio_threshold', 0.33)
                    cv2.putText(frame, f"HeightRatio: {height_ratio:.3f} (<= {threshold})",
                               (x + 5, debug_y + line_height * 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
                    cv2.putText(frame, f"ValidKP:     {debug_info.get('valid_kp_count', 0)}",
                               (x + 5, debug_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

                    # 최종 신뢰도
                    cv2.putText(frame, f"CONF:        {raw_conf:.3f}",
                               (x + 5, debug_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, conf_color, 1)

                else:
                    # 기타 모드
                    cv2.rectangle(frame, (x, debug_y - 5), (x + 150, debug_y + line_height * 2 + 5),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, f"MODE: {mode}",
                               (x + 5, debug_y + line_height * 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
                    cv2.putText(frame, f"CONF: {raw_conf:.3f}",
                               (x + 5, debug_y + line_height * 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, conf_color, 1)

        # 싸움 정보 표시
        if is_fighting:
            cv2.putText(frame,
                       f"FIGHTING DETECTED! ({fight_confidence:.2f})",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (0, 0, 255),
                       3)
            # 프레임 전체에 경고 오버레이
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    def release(self):
        """리소스 해제"""
        # YOLO 모델은 자동으로 관리되므로 특별한 해제 작업 불필요
        pass

