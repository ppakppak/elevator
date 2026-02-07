"""
천장 카메라 최적화 앙상블 낙상 감지 모듈
엘리베이터 등 천장에 설치된 카메라에서의 낙상 감지에 최적화

앙상블 구성:
1. BBox 형태 분석 (aspect ratio, area change)
2. 움직임 분석 (optical flow 기반 수직 이동)
3. 머리 위치 추적 (head position tracking)
4. 키포인트 기반 (가능한 경우)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class FallDetectionResult:
    """낙상 감지 결과"""
    is_fallen: bool = False
    confidence: float = 0.0

    # 개별 감지기 점수
    bbox_score: float = 0.0
    motion_score: float = 0.0
    head_score: float = 0.0
    keypoint_score: float = 0.0

    # 디버그 정보
    debug_info: Dict = field(default_factory=dict)


class BBoxAnalyzer:
    """바운딩 박스 형태 분석 기반 낙상 감지"""

    def __init__(self):
        # 이전 bbox 저장 (person_id -> deque of (w, h, area))
        self.bbox_history: Dict[int, deque] = {}
        self.HISTORY_SIZE = 10

        # 천장 카메라 임계값 (오탐 감소를 위해 더 엄격하게)
        self.ASPECT_THRESHOLD_LOW = 0.65  # 이보다 낮으면 확실히 누움
        self.ASPECT_THRESHOLD_HIGH = 1.2  # 이보다 높으면 서 있음
        self.AREA_CHANGE_THRESHOLD = 1.3  # 면적 30% 이상 증가시 낙상 의심

    def analyze(self, person_id: int, bbox: Tuple[int, int, int, int],
                frame_height: int) -> Tuple[float, Dict]:
        """
        BBox 분석으로 낙상 점수 계산

        Returns:
            (score, debug_info): 0.0~1.0 점수와 디버그 정보
        """
        x, y, w, h = bbox
        debug_info = {}

        # 히스토리 초기화
        if person_id not in self.bbox_history:
            self.bbox_history[person_id] = deque(maxlen=self.HISTORY_SIZE)

        area = w * h
        aspect_ratio = h / (w + 1e-6)

        # 히스토리에 추가
        self.bbox_history[person_id].append({
            'w': w, 'h': h, 'area': area, 'aspect': aspect_ratio, 'y': y
        })

        history = self.bbox_history[person_id]

        # === 1. 종횡비 분석 (천장 카메라용) ===
        # 천장 카메라에서는 서 있어도 aspect ratio가 낮을 수 있음
        aspect_score = 0.0
        if aspect_ratio < self.ASPECT_THRESHOLD_LOW:
            # 확실히 누운 상태
            aspect_score = 1.0
        elif aspect_ratio < self.ASPECT_THRESHOLD_HIGH:
            # 애매한 구간 - 선형 보간
            aspect_score = 1.0 - (aspect_ratio - self.ASPECT_THRESHOLD_LOW) / \
                          (self.ASPECT_THRESHOLD_HIGH - self.ASPECT_THRESHOLD_LOW)

        debug_info['aspect_ratio'] = aspect_ratio
        debug_info['aspect_score'] = aspect_score

        # === 2. 면적 변화 분석 ===
        area_change_score = 0.0
        if len(history) >= 3:
            # 최근 3프레임 전과 비교
            prev_area = history[-3]['area']
            if prev_area > 0:
                area_ratio = area / prev_area
                if area_ratio > self.AREA_CHANGE_THRESHOLD:
                    # 면적이 급격히 증가 = 쓰러짐
                    area_change_score = min((area_ratio - 1.0) / 0.5, 1.0)

        debug_info['area'] = area
        debug_info['area_change_score'] = area_change_score

        # === 3. bbox 상단(머리) Y 위치 ===
        # 머리가 화면 아래쪽에 있으면 쓰러진 것
        head_y_ratio = y / (frame_height + 1e-6)
        head_position_score = 0.0
        if head_y_ratio > 0.55:  # 화면 55% 아래
            head_position_score = min((head_y_ratio - 0.55) / 0.25, 1.0)

        debug_info['head_y_ratio'] = head_y_ratio
        debug_info['head_position_score'] = head_position_score

        # === 4. bbox 높이 급격한 감소 ===
        height_drop_score = 0.0
        if len(history) >= 5:
            prev_heights = [h['h'] for h in list(history)[-5:-1]]
            avg_prev_height = np.mean(prev_heights)
            if avg_prev_height > 0:
                height_ratio = h / avg_prev_height
                if height_ratio < 0.6:  # 높이가 40% 이상 감소
                    height_drop_score = 1.0 - height_ratio

        debug_info['height_drop_score'] = height_drop_score

        # === 5. bbox 너비가 높이보다 큰 경우 (누워있음) ===
        width_dominant_score = 0.0
        if w > h * 1.2:  # 너비가 높이의 1.2배 이상
            width_dominant_score = min((w / h - 1.0) / 1.0, 1.0)

        debug_info['width_dominant_score'] = width_dominant_score

        # === 종합 점수 (천장 카메라 최적화) ===
        # 종횡비가 가장 중요한 지표
        final_score = (
            aspect_score * 0.50 +          # 종횡비 (핵심)
            width_dominant_score * 0.20 +  # 너비 우세
            area_change_score * 0.15 +     # 면적 변화
            height_drop_score * 0.15       # 높이 감소
            # head_position_score 제거 - 천장 카메라에서는 부정확
        )

        debug_info['bbox_final_score'] = final_score

        return final_score, debug_info

    def reset(self, person_id: int = None):
        """히스토리 리셋"""
        if person_id is not None:
            self.bbox_history.pop(person_id, None)
        else:
            self.bbox_history.clear()


class MotionAnalyzer:
    """움직임(옵티컬 플로우) 기반 낙상 감지"""

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.motion_history: Dict[int, deque] = {}
        self.HISTORY_SIZE = 15

        # 임계값
        self.VERTICAL_MOTION_THRESHOLD = 3.0  # 수직 이동 임계값
        self.SUDDEN_STOP_THRESHOLD = 0.5      # 급정지 임계값

    def analyze(self, frame: np.ndarray, person_id: int,
                bbox: Tuple[int, int, int, int]) -> Tuple[float, Dict]:
        """
        움직임 분석으로 낙상 점수 계산

        Args:
            frame: 현재 프레임 (BGR)
            person_id: 사람 ID
            bbox: (x, y, w, h)

        Returns:
            (score, debug_info)
        """
        debug_info = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if person_id not in self.motion_history:
            self.motion_history[person_id] = deque(maxlen=self.HISTORY_SIZE)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, {'status': 'initializing'}

        x, y, w, h = bbox
        # bbox 영역을 약간 확장 (패딩)
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return 0.0, {'status': 'bbox_too_small'}

        try:
            roi_curr = gray[y1:y2, x1:x2]
            roi_prev = self.prev_gray[y1:y2, x1:x2]

            if roi_curr.shape != roi_prev.shape:
                self.prev_gray = gray
                return 0.0, {'status': 'shape_mismatch'}

            # 옵티컬 플로우 계산
            flow = cv2.calcOpticalFlowFarneback(
                roi_prev, roi_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # 움직임 분석
            flow_x = flow[:, :, 0]  # 수평 움직임
            flow_y = flow[:, :, 1]  # 수직 움직임

            avg_flow_x = np.mean(flow_x)
            avg_flow_y = np.mean(flow_y)
            flow_magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)

            # 히스토리에 저장
            self.motion_history[person_id].append({
                'flow_x': avg_flow_x,
                'flow_y': avg_flow_y,
                'magnitude': flow_magnitude
            })

            debug_info['flow_x'] = avg_flow_x
            debug_info['flow_y'] = avg_flow_y
            debug_info['magnitude'] = flow_magnitude

        except Exception as e:
            self.prev_gray = gray
            return 0.0, {'status': f'error: {str(e)}'}

        self.prev_gray = gray

        # === 1. 수직 하강 움직임 (아래로 떨어짐) ===
        # 천장 카메라에서 Y 증가 = 아래로 이동
        vertical_score = 0.0
        if avg_flow_y > self.VERTICAL_MOTION_THRESHOLD:
            vertical_score = min(avg_flow_y / 8.0, 1.0)

        debug_info['vertical_score'] = vertical_score

        # === 2. 급정지 감지 (쓰러진 후 움직임 멈춤) ===
        stop_score = 0.0
        history = list(self.motion_history[person_id])
        if len(history) >= 5:
            # 이전에 큰 움직임이 있었고, 지금은 멈춤
            prev_magnitudes = [h['magnitude'] for h in history[-5:-2]]
            recent_magnitudes = [h['magnitude'] for h in history[-2:]]

            avg_prev_mag = np.mean(prev_magnitudes)
            avg_recent_mag = np.mean(recent_magnitudes)

            if avg_prev_mag > 3.0 and avg_recent_mag < self.SUDDEN_STOP_THRESHOLD:
                # 큰 움직임 후 급정지 = 쓰러짐
                stop_score = min((avg_prev_mag - avg_recent_mag) / 5.0, 1.0)

        debug_info['stop_score'] = stop_score

        # === 3. 누적 수직 이동량 ===
        cumulative_score = 0.0
        if len(history) >= 5:
            recent_y_flows = [h['flow_y'] for h in history[-5:]]
            cumulative_y = sum(recent_y_flows)
            if cumulative_y > 10.0:  # 누적 하강
                cumulative_score = min(cumulative_y / 20.0, 1.0)

        debug_info['cumulative_score'] = cumulative_score

        # === 종합 점수 ===
        final_score = (
            vertical_score * 0.40 +
            stop_score * 0.35 +
            cumulative_score * 0.25
        )

        debug_info['motion_final_score'] = final_score

        return final_score, debug_info

    def reset(self):
        """상태 리셋"""
        self.prev_gray = None
        self.motion_history.clear()


class HeadTracker:
    """머리 위치 추적 기반 낙상 감지"""

    def __init__(self):
        self.head_history: Dict[int, deque] = {}
        self.HISTORY_SIZE = 20

        # 임계값
        self.FALL_SPEED_THRESHOLD = 0.03    # 프레임당 Y 변화율
        self.LOW_POSITION_THRESHOLD = 0.60  # 화면 60% 아래면 쓰러진 것

    def analyze(self, person_id: int, head_y: float,
                frame_height: int) -> Tuple[float, Dict]:
        """
        머리 위치 추적으로 낙상 점수 계산

        Args:
            person_id: 사람 ID
            head_y: 머리 Y 좌표 (픽셀)
            frame_height: 프레임 높이

        Returns:
            (score, debug_info)
        """
        debug_info = {}

        if person_id not in self.head_history:
            self.head_history[person_id] = deque(maxlen=self.HISTORY_SIZE)

        # 정규화된 Y 위치 (0=상단, 1=하단)
        normalized_y = head_y / (frame_height + 1e-6)
        self.head_history[person_id].append(normalized_y)

        history = list(self.head_history[person_id])
        debug_info['normalized_y'] = normalized_y
        debug_info['history_length'] = len(history)

        if len(history) < 3:
            return 0.0, debug_info

        # === 1. 급격한 하강 속도 ===
        speed_score = 0.0
        if len(history) >= 5:
            recent_positions = history[-5:]
            y_velocity = recent_positions[-1] - recent_positions[0]

            # 양수 = 아래로 이동
            if y_velocity > self.FALL_SPEED_THRESHOLD * 5:
                speed_score = min(y_velocity / 0.2, 1.0)

        debug_info['y_velocity'] = y_velocity if len(history) >= 5 else 0
        debug_info['speed_score'] = speed_score

        # === 2. 현재 위치 (낮은 위치 = 쓰러짐) ===
        position_score = 0.0
        if normalized_y > self.LOW_POSITION_THRESHOLD:
            position_score = (normalized_y - self.LOW_POSITION_THRESHOLD) / 0.3
            position_score = min(position_score, 1.0)

        debug_info['position_score'] = position_score

        # === 3. 위치 안정성 (쓰러진 후 움직임 적음) ===
        stability_score = 0.0
        if len(history) >= 10:
            recent = history[-5:]
            std = np.std(recent)

            # 낮은 위치에서 안정적 = 쓰러진 상태
            if normalized_y > 0.5 and std < 0.02:
                stability_score = (1.0 - std / 0.02) * (normalized_y - 0.5) / 0.5
                stability_score = min(max(stability_score, 0.0), 1.0)

        debug_info['stability_score'] = stability_score

        # === 4. 최고점 대비 하락 ===
        drop_score = 0.0
        if len(history) >= 10:
            max_y = min(history[-10:])  # 가장 높았던 위치 (Y가 작을수록 높음)
            drop = normalized_y - max_y
            if drop > 0.15:  # 15% 이상 하락
                drop_score = min(drop / 0.3, 1.0)

        debug_info['drop_score'] = drop_score

        # === 종합 점수 ===
        final_score = (
            speed_score * 0.30 +
            position_score * 0.35 +
            stability_score * 0.15 +
            drop_score * 0.20
        )

        debug_info['head_final_score'] = final_score

        return final_score, debug_info

    def reset(self, person_id: int = None):
        """히스토리 리셋"""
        if person_id is not None:
            self.head_history.pop(person_id, None)
        else:
            self.head_history.clear()


class KeypointAnalyzer:
    """키포인트 기반 낙상 감지 (가능한 경우)"""

    # COCO 키포인트 인덱스
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self):
        self.MIN_CONFIDENCE = 0.2  # 천장 카메라용으로 더 낮춤

    def analyze(self, keypoints: np.ndarray,
                frame_width: int, frame_height: int) -> Tuple[float, Dict]:
        """
        키포인트 분석으로 낙상 점수 계산

        Args:
            keypoints: [17, 3] 형태의 키포인트 배열 (x, y, confidence)

        Returns:
            (score, debug_info)
        """
        debug_info = {'mode': 'keypoint'}

        if keypoints is None or len(keypoints) < 17:
            return 0.0, {'status': 'no_keypoints'}

        # 유효한 키포인트 추출
        valid_kps = []
        for idx, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > self.MIN_CONFIDENCE:
                valid_kps.append((idx, kp[0], kp[1], kp[2]))

        debug_info['valid_kp_count'] = len(valid_kps)

        if len(valid_kps) < 3:
            return 0.0, debug_info

        # 상체 키포인트 추출
        def get_kp(idx):
            for i, x, y, c in valid_kps:
                if i == idx:
                    return (x, y, c)
            return None

        nose = get_kp(self.NOSE)
        left_eye = get_kp(self.LEFT_EYE)
        right_eye = get_kp(self.RIGHT_EYE)
        left_shoulder = get_kp(self.LEFT_SHOULDER)
        right_shoulder = get_kp(self.RIGHT_SHOULDER)
        left_hip = get_kp(self.LEFT_HIP)
        right_hip = get_kp(self.RIGHT_HIP)

        # === 1. 어깨 기울기 ===
        shoulder_tilt_score = 0.0
        if left_shoulder and right_shoulder:
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            shoulder_angle = abs(np.arctan2(dy, dx) * 180 / np.pi)

            # 수평(0도)에서 벗어날수록 쓰러짐
            if shoulder_angle > 30:
                shoulder_tilt_score = min((shoulder_angle - 30) / 40, 1.0)

            debug_info['shoulder_angle'] = shoulder_angle

        debug_info['shoulder_tilt_score'] = shoulder_tilt_score

        # === 2. 머리-어깨 수직 거리 ===
        head_shoulder_score = 0.0
        head_point = nose or left_eye or right_eye

        if head_point and (left_shoulder or right_shoulder):
            shoulder_y = left_shoulder[1] if left_shoulder else right_shoulder[1]
            head_y = head_point[1]

            # 정상: 머리가 어깨 위 (Y가 작음)
            # 쓰러짐: 머리와 어깨가 비슷하거나 머리가 아래
            vertical_diff = (shoulder_y - head_y) / frame_height

            if vertical_diff < 0.05:  # 머리가 어깨와 거의 같은 높이
                head_shoulder_score = 1.0 - (vertical_diff / 0.05)
                head_shoulder_score = max(0.0, min(1.0, head_shoulder_score))

        debug_info['head_shoulder_score'] = head_shoulder_score

        # === 3. 키포인트 분산 (가로 vs 세로) ===
        spread_score = 0.0
        valid_x = [kp[1] for kp in valid_kps]
        valid_y = [kp[2] for kp in valid_kps]

        x_range = max(valid_x) - min(valid_x)
        y_range = max(valid_y) - min(valid_y)

        spread_ratio = 0
        if y_range > 0:
            spread_ratio = x_range / (y_range + 1e-6)
            # 가로로 넓게 퍼짐 = 쓰러짐 (천장 카메라에서 더 민감하게)
            if spread_ratio > 1.0:
                spread_score = min((spread_ratio - 1.0) / 1.0, 1.0)

        debug_info['spread_ratio'] = spread_ratio
        debug_info['spread_score'] = spread_score

        # === 4. 골반 기울기 (추가) ===
        hip_tilt_score = 0.0
        if left_hip and right_hip:
            dx = right_hip[0] - left_hip[0]
            dy = right_hip[1] - left_hip[1]
            hip_angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if hip_angle > 25:
                hip_tilt_score = min((hip_angle - 25) / 35, 1.0)
            debug_info['hip_angle'] = hip_angle
        debug_info['hip_tilt_score'] = hip_tilt_score

        # === 종합 점수 (천장 카메라 최적화) ===
        final_score = (
            shoulder_tilt_score * 0.30 +
            spread_score * 0.35 +           # 분산이 더 중요
            hip_tilt_score * 0.20 +
            head_shoulder_score * 0.15
        )

        debug_info['keypoint_final_score'] = final_score

        return final_score, debug_info


class CeilingFallDetector:
    """
    천장 카메라 최적화 앙상블 낙상 감지기

    4가지 감지 방법을 앙상블:
    1. BBox 형태 분석
    2. 움직임 분석 (Optical Flow) - 선택적
    3. 머리 위치 추적
    4. 키포인트 분석 (가능한 경우)
    """

    def __init__(self,
                 fall_confirm_frames: int = 3,
                 confidence_threshold: float = 0.5,
                 enable_motion: bool = False):
        """
        Args:
            fall_confirm_frames: 낙상 확정에 필요한 연속 프레임 수
            confidence_threshold: 낙상 판정 신뢰도 임계값
            enable_motion: 옵티컬 플로우 활성화 (느림, 기본값 False)
        """
        # 개별 분석기
        self.bbox_analyzer = BBoxAnalyzer()
        self.enable_motion = enable_motion
        self.motion_analyzer = MotionAnalyzer() if enable_motion else None
        self.head_tracker = HeadTracker()
        self.keypoint_analyzer = KeypointAnalyzer()

        # 앙상블 가중치 (천장 카메라 최적화)
        # BBox 형태가 가장 신뢰성 높음
        if enable_motion:
            self.weights = {
                'bbox': 0.50,      # 종횡비/형태 분석 (핵심)
                'motion': 0.20,    # 움직임 분석
                'head': 0.15,      # 머리 위치 (천장에서는 덜 신뢰)
                'keypoint': 0.15   # 키포인트 분석
            }
        else:
            # 옵티컬 플로우 비활성화 시 가중치 재분배
            self.weights = {
                'bbox': 0.60,      # bbox에 더 많은 가중치
                'motion': 0.00,    # 비활성화
                'head': 0.20,
                'keypoint': 0.20
            }

        # 시간 필터링
        self.fall_confirm_frames = 3
        self.fall_frame_counts: Dict[int, int] = {}

        # Hysteresis 기반 이중 임계값 (오탐 감소 + 지속 감지)
        self.entry_threshold = 0.40    # 낙상 진입 임계값 (240 프레임 감지용)
        self.exit_threshold = 0.22     # 낙상 유지 임계값
        self.confidence_threshold = self.entry_threshold  # 호환성

        # 낙상 상태 지속 (한번 감지되면 유지)
        self.fall_state: Dict[int, bool] = {}              # 현재 낙상 상태
        self.fall_persistence_frames: Dict[int, int] = {}  # 낙상 지속 프레임 수
        self.low_conf_frames: Dict[int, int] = {}          # 연속 낮은 신뢰도 프레임 수
        self.max_persistence_frames = 400  # 최대 blind 지속 (~13초 at 30fps)
        self.reset_after_low_frames = 15   # 낮은 신뢰도 15프레임 연속시 리셋

        # 디버그 모드
        self.debug_mode = False
        self.last_debug_info: Dict[int, Dict] = {}

    def set_weights(self, bbox: float = None, motion: float = None,
                    head: float = None, keypoint: float = None):
        """앙상블 가중치 조정"""
        if bbox is not None:
            self.weights['bbox'] = bbox
        if motion is not None:
            self.weights['motion'] = motion
        if head is not None:
            self.weights['head'] = head
        if keypoint is not None:
            self.weights['keypoint'] = keypoint

        # 정규화
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total

    def detect(self, frame: np.ndarray, person_id: int,
               bbox: Tuple[int, int, int, int],
               keypoints: np.ndarray = None) -> FallDetectionResult:
        """
        앙상블 낙상 감지 수행

        Args:
            frame: 현재 프레임 (BGR)
            person_id: 사람 ID
            bbox: (x, y, w, h) 바운딩 박스
            keypoints: [17, 3] 키포인트 배열 (선택)

        Returns:
            FallDetectionResult: 감지 결과
        """
        height, width = frame.shape[:2]
        x, y, w, h = bbox

        result = FallDetectionResult()
        debug_info = {'person_id': person_id}

        # === 1. BBox 분석 ===
        bbox_score, bbox_debug = self.bbox_analyzer.analyze(
            person_id, bbox, height
        )
        result.bbox_score = bbox_score
        debug_info['bbox'] = bbox_debug

        # === 2. 움직임 분석 (선택적) ===
        motion_score = 0.0
        motion_debug = {'status': 'disabled'}
        if self.enable_motion and self.motion_analyzer is not None:
            motion_score, motion_debug = self.motion_analyzer.analyze(
                frame, person_id, bbox
            )
        result.motion_score = motion_score
        debug_info['motion'] = motion_debug

        # === 3. 머리 위치 추적 ===
        # 머리 Y 좌표 추정 (bbox 상단 또는 키포인트)
        head_y = y  # 기본: bbox 상단
        if keypoints is not None and len(keypoints) >= 3:
            # 코 또는 눈 위치 사용
            for idx in [0, 1, 2]:  # nose, left_eye, right_eye
                if keypoints[idx][2] > 0.3:
                    head_y = keypoints[idx][1]
                    break

        head_score, head_debug = self.head_tracker.analyze(
            person_id, head_y, height
        )
        result.head_score = head_score
        debug_info['head'] = head_debug

        # === 4. 키포인트 분석 (가능한 경우) ===
        keypoint_score = 0.0
        keypoint_debug = {'status': 'skipped'}

        if keypoints is not None:
            keypoint_score, keypoint_debug = self.keypoint_analyzer.analyze(
                keypoints, width, height
            )

        result.keypoint_score = keypoint_score
        debug_info['keypoint'] = keypoint_debug

        # === 앙상블 점수 계산 ===
        # 키포인트가 유효하지 않으면 다른 방법에 가중치 재분배
        if keypoint_score == 0.0 and keypoint_debug.get('status') in ['no_keypoints', 'skipped']:
            # 키포인트 없이 앙상블
            effective_weights = {
                'bbox': self.weights['bbox'] + self.weights['keypoint'] * 0.4,
                'motion': self.weights['motion'] + self.weights['keypoint'] * 0.3,
                'head': self.weights['head'] + self.weights['keypoint'] * 0.3,
            }
            raw_confidence = (
                bbox_score * effective_weights['bbox'] +
                motion_score * effective_weights['motion'] +
                head_score * effective_weights['head']
            )
        else:
            # 전체 앙상블
            raw_confidence = (
                bbox_score * self.weights['bbox'] +
                motion_score * self.weights['motion'] +
                head_score * self.weights['head'] +
                keypoint_score * self.weights['keypoint']
            )

        # === 즉시 낙상 판정 (높은 확신도) ===
        # BBox 점수가 높으면 시간 필터링 없이 즉시 판정
        instant_fall = False
        if bbox_score >= 0.70:  # 종횡비가 확실히 낮은 경우
            instant_fall = True
            raw_confidence = max(raw_confidence, 0.65)

        debug_info['raw_confidence'] = raw_confidence
        debug_info['weights'] = self.weights.copy()
        debug_info['instant_fall'] = instant_fall

        # === Hysteresis 기반 낙상 판정 ===
        # 현재 낙상 상태 확인
        currently_fallen = self.fall_state.get(person_id, False)
        persistence_count = self.fall_persistence_frames.get(person_id, 0)

        # 프레임 카운터 업데이트
        if raw_confidence > self.entry_threshold:
            self.fall_frame_counts[person_id] = \
                self.fall_frame_counts.get(person_id, 0) + 1
        elif raw_confidence < self.exit_threshold:
            # exit_threshold 미만이면 카운터 감소
            self.fall_frame_counts[person_id] = max(0,
                self.fall_frame_counts.get(person_id, 0) - 1)
        # else: exit과 entry 사이는 카운터 유지

        frame_count = self.fall_frame_counts.get(person_id, 0)

        # 낮은 신뢰도 카운터 관리
        low_conf_count = self.low_conf_frames.get(person_id, 0)
        if raw_confidence < self.exit_threshold:
            self.low_conf_frames[person_id] = low_conf_count + 1
        else:
            self.low_conf_frames[person_id] = 0
        low_conf_count = self.low_conf_frames[person_id]

        # 낙상 상태 결정 (Hysteresis)
        if instant_fall:
            # 즉시 낙상
            is_fallen = True
            self.fall_state[person_id] = True
            self.fall_persistence_frames[person_id] = 0
            self.low_conf_frames[person_id] = 0
            confidence = raw_confidence
        elif currently_fallen:
            # 이미 낙상 상태 - 유지 여부 결정
            # 연속으로 낮은 신뢰도가 N프레임 지속되면 낙상 해제
            if low_conf_count >= self.reset_after_low_frames:
                # 낮은 신뢰도가 오래 지속 - 낙상 해제 (일어남)
                is_fallen = False
                self.fall_state[person_id] = False
                self.fall_persistence_frames[person_id] = 0
                confidence = raw_confidence
            elif raw_confidence >= self.exit_threshold:
                # exit_threshold 이상이면 낙상 상태 유지
                is_fallen = True
                self.fall_persistence_frames[person_id] = persistence_count + 1
                confidence = max(raw_confidence, 0.35)
            elif persistence_count < self.max_persistence_frames:
                # 잠시 점수가 떨어져도 일정 기간 유지
                is_fallen = True
                self.fall_persistence_frames[person_id] = persistence_count + 1
                confidence = max(raw_confidence, 0.30)
            else:
                # 최대 blind 지속 시간 초과 - 낙상 해제
                is_fallen = False
                self.fall_state[person_id] = False
                self.fall_persistence_frames[person_id] = 0
                confidence = raw_confidence
        else:
            # 새로운 낙상 감지
            if frame_count >= self.fall_confirm_frames:
                is_fallen = True
                self.fall_state[person_id] = True
                self.fall_persistence_frames[person_id] = 0
                confidence = raw_confidence
            else:
                is_fallen = False
                frame_ratio = frame_count / self.fall_confirm_frames
                confidence = raw_confidence * frame_ratio

        debug_info['currently_fallen'] = currently_fallen
        debug_info['persistence_count'] = persistence_count
        debug_info['entry_threshold'] = self.entry_threshold
        debug_info['exit_threshold'] = self.exit_threshold

        result.is_fallen = is_fallen
        result.confidence = confidence
        result.debug_info = debug_info

        # 디버그 정보 저장
        self.last_debug_info[person_id] = debug_info

        # 디버그 출력
        if self.debug_mode and person_id == 0:
            print(f"[Ceiling-Ensemble] BBox:{bbox_score:.2f} Motion:{motion_score:.2f} "
                  f"Head:{head_score:.2f} KP:{keypoint_score:.2f} -> Raw:{raw_confidence:.3f} "
                  f"Frames:{frame_count}/{self.fall_confirm_frames} Fallen:{is_fallen}")

        return result

    def reset(self, person_id: int = None):
        """상태 리셋"""
        self.bbox_analyzer.reset(person_id)
        if self.motion_analyzer:
            self.motion_analyzer.reset()
        self.head_tracker.reset(person_id)

        if person_id is not None:
            self.fall_frame_counts.pop(person_id, None)
            self.fall_state.pop(person_id, None)
            self.fall_persistence_frames.pop(person_id, None)
            self.low_conf_frames.pop(person_id, None)
            self.last_debug_info.pop(person_id, None)
        else:
            self.fall_frame_counts.clear()
            self.fall_state.clear()
            self.fall_persistence_frames.clear()
            self.low_conf_frames.clear()
            self.last_debug_info.clear()

    def get_debug_overlay(self, frame: np.ndarray, person_id: int,
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """디버그 정보를 프레임에 오버레이"""
        if person_id not in self.last_debug_info:
            return frame

        debug = self.last_debug_info[person_id]
        x, y, w, h = bbox

        # 배경 박스
        overlay_x = x
        overlay_y = y + h + 10
        overlay_h = 120
        overlay_w = 280

        cv2.rectangle(frame,
                     (overlay_x, overlay_y),
                     (overlay_x + overlay_w, overlay_y + overlay_h),
                     (0, 0, 0), -1)
        cv2.rectangle(frame,
                     (overlay_x, overlay_y),
                     (overlay_x + overlay_w, overlay_y + overlay_h),
                     (100, 100, 100), 1)

        # 텍스트 출력
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (200, 200, 200)
        line_height = 16

        lines = [
            f"=== Ceiling Ensemble ===",
            f"BBox:    {debug.get('bbox', {}).get('bbox_final_score', 0):.3f} (w:{self.weights['bbox']:.2f})",
            f"Motion:  {debug.get('motion', {}).get('motion_final_score', 0):.3f} (w:{self.weights['motion']:.2f})",
            f"Head:    {debug.get('head', {}).get('head_final_score', 0):.3f} (w:{self.weights['head']:.2f})",
            f"KeyPt:   {debug.get('keypoint', {}).get('keypoint_final_score', 0):.3f} (w:{self.weights['keypoint']:.2f})",
            f"------------------------",
            f"Raw Conf: {debug.get('raw_confidence', 0):.3f}",
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                       (overlay_x + 5, overlay_y + 15 + i * line_height),
                       font, font_scale, color, 1)

        return frame


# 테스트용 메인
if __name__ == "__main__":
    import sys

    print("천장 카메라 앙상블 낙상 감지기 테스트")

    # 감지기 생성
    detector = CeilingFallDetector(fall_confirm_frames=3)
    detector.debug_mode = True

    # 테스트 영상
    video_path = "videos/승강기1.mp4" if len(sys.argv) < 2 else sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {video_path}")
        sys.exit(1)

    print(f"영상 로드: {video_path}")
    print("Press 'q' to quit, 'd' to toggle debug mode")

    # YOLO 모델 로드 (사람 감지용)
    try:
        from ultralytics import YOLO
        yolo = YOLO('yolo11n-pose.pt')
        print("YOLO 모델 로드 완료")
    except Exception as e:
        print(f"YOLO 로드 실패: {e}")
        sys.exit(1)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO로 사람 감지
        results = yolo(frame, conf=0.5, verbose=False)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints_data = None
            if results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                kps = keypoints_data[i] if keypoints_data is not None else None

                # 낙상 감지
                result = detector.detect(frame, i, bbox, kps)

                # 결과 시각화
                color = (0, 0, 255) if result.is_fallen else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{'FALLEN' if result.is_fallen else 'OK'} ({result.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 디버그 오버레이
                if detector.debug_mode:
                    frame = detector.get_debug_overlay(frame, i, bbox)

        # 프레임 정보
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Ceiling Fall Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            detector.debug_mode = not detector.debug_mode
            print(f"Debug mode: {detector.debug_mode}")

    cap.release()
    cv2.destroyAllWindows()
