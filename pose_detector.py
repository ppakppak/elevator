"""
MediaPipe 기반 포즈 추정 모듈
쓰러짐 및 싸움 감지 기능 포함
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PersonPose:
    """한 사람의 포즈 정보"""
    landmarks: List
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]
    is_fallen: bool = False
    fall_confidence: float = 0.0


class PoseDetector:
    """MediaPipe를 사용한 포즈 추정 및 이벤트 감지 클래스"""
    
    def __init__(self, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_complexity=1,
                 enable_multi_person=False,
                 grid_rows=2,
                 grid_cols=2):
        """
        Args:
            min_detection_confidence: 최소 감지 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
            model_complexity: 모델 복잡도 (0, 1, 2)
            enable_multi_person: 다중 인물 감지 활성화 (프레임을 격자로 나눔)
            grid_rows: 격자 행 수 (다중 인물 감지 시)
            grid_cols: 격자 열 수 (다중 인물 감지 시)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        self.enable_multi_person = enable_multi_person
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # 포즈 랜드마크 인덱스 (MediaPipe Pose)
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.NOSE = 0
        
        # 쓰러짐 감지 임계값 (상부 카메라/탑뷰 최적화)
        self.ASPECT_RATIO_THRESHOLD = 2.0  # 종횡비: 서있음 > 2.0, 쓰러짐/앉음 < 1.5
        self.SPREAD_RATIO_THRESHOLD = 0.5  # 분산비: 서있음 < 0.5, 쓰러짐 > 0.8
        self.BODY_LENGTH_THRESHOLD = 0.4   # 신체길이: 서있음 > 0.4, 쓰러짐 < 0.25, 앉음 < 0.2
        self.BBOX_AREA_THRESHOLD = 0.05    # 바운딩박스 면적 기준

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
    
    def get_landmark_point(self, landmarks, idx: int) -> Optional[Tuple[float, float]]:
        """랜드마크 포인트 가져오기"""
        if landmarks and idx < len(landmarks.landmark):
            landmark = landmarks.landmark[idx]
            return (landmark.x, landmark.y)
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
    
    def detect_fall(self, landmarks, normalized_coords: Optional[dict] = None,
                   person_id: int = 0) -> Tuple[bool, float, dict]:
        """
        쓰러짐 감지 (상부 카메라/탑뷰 최적화)
        쓰러짐과 쪼그려 앉음 모두 감지

        Args:
            landmarks: MediaPipe 랜드마크 객체
            normalized_coords: 정규화된 좌표 딕셔너리 (다중 인물 감지 시 변환된 좌표)
            person_id: 사람 ID (시간 필터링용)

        Returns:
            (is_fallen, confidence, debug_info): 쓰러짐 여부, 신뢰도, 디버그 정보
        """
        debug_info = {'aspect': 0.0, 'spread': 0.0, 'length': 0.0, 'area': 0.0, 'raw_conf': 0.0}

        if not landmarks:
            return False, 0.0, debug_info

        # 좌표 추출 (정규화된 좌표 0-1 범위)
        if normalized_coords:
            coords = normalized_coords
        else:
            coords = {}
            for idx, lm in enumerate(landmarks.landmark):
                coords[idx] = (lm.x, lm.y)

        # 유효한 좌표만 필터링 (visibility 기반)
        valid_coords = []
        for idx, lm in enumerate(landmarks.landmark):
            if lm.visibility > 0.5:
                if normalized_coords:
                    valid_coords.append(normalized_coords.get(idx, (lm.x, lm.y)))
                else:
                    valid_coords.append((lm.x, lm.y))

        if len(valid_coords) < 4:
            return False, 0.0, debug_info

        valid_coords = np.array(valid_coords)

        # === 1. 바운딩 박스 종횡비 계산 (30%) ===
        min_x = np.min(valid_coords[:, 0])
        max_x = np.max(valid_coords[:, 0])
        min_y = np.min(valid_coords[:, 1])
        max_y = np.max(valid_coords[:, 1])

        bbox_width = max_x - min_x + 1e-6
        bbox_height = max_y - min_y + 1e-6
        aspect_ratio = bbox_height / bbox_width

        if aspect_ratio < self.ASPECT_RATIO_THRESHOLD:
            aspect_score = 1.0 - (aspect_ratio / self.ASPECT_RATIO_THRESHOLD)
        else:
            aspect_score = 0.0
        aspect_score = max(0.0, min(1.0, aspect_score))

        # === 2. 키포인트 분산 비율 계산 (25%) ===
        x_std = np.std(valid_coords[:, 0])
        y_std = np.std(valid_coords[:, 1]) + 1e-6
        spread_ratio = x_std / y_std

        if spread_ratio > self.SPREAD_RATIO_THRESHOLD:
            spread_score = min((spread_ratio - self.SPREAD_RATIO_THRESHOLD) / 0.5, 1.0)
        else:
            spread_score = 0.0
        spread_score = max(0.0, min(1.0, spread_score))

        # === 3. 머리-발목 거리 계산 (30%) - 핵심 지표 ===
        nose = coords.get(self.NOSE)
        left_ankle = coords.get(self.LEFT_ANKLE)
        right_ankle = coords.get(self.RIGHT_ANKLE)
        left_hip = coords.get(self.LEFT_HIP)
        right_hip = coords.get(self.RIGHT_HIP)

        length_score = 0.0
        body_length_normalized = 0.0

        if nose is not None:
            # 발목 중심 계산
            if left_ankle is not None and right_ankle is not None:
                foot_center = ((left_ankle[0] + right_ankle[0]) / 2,
                              (left_ankle[1] + right_ankle[1]) / 2)
            elif left_ankle is not None:
                foot_center = left_ankle
            elif right_ankle is not None:
                foot_center = right_ankle
            elif left_hip is not None and right_hip is not None:
                foot_center = ((left_hip[0] + right_hip[0]) / 2,
                              (left_hip[1] + right_hip[1]) / 2)
            else:
                foot_center = None

            if foot_center is not None:
                body_length = np.sqrt((nose[0] - foot_center[0])**2 +
                                     (nose[1] - foot_center[1])**2)
                body_length_normalized = body_length  # 이미 0-1 범위

                if body_length_normalized < self.BODY_LENGTH_THRESHOLD:
                    length_score = 1.0 - (body_length_normalized / self.BODY_LENGTH_THRESHOLD)
                else:
                    length_score = 0.0
                length_score = max(0.0, min(1.0, length_score))

        # === 4. 바운딩 박스 면적 계산 (15%) ===
        bbox_area = bbox_width * bbox_height

        if bbox_area < self.BBOX_AREA_THRESHOLD:
            area_score = 1.0 - (bbox_area / self.BBOX_AREA_THRESHOLD)
        else:
            area_score = 0.0
        area_score = max(0.0, min(1.0, area_score))

        # === 종합 신뢰도 계산 ===
        raw_confidence = (aspect_score * 0.30 +
                         spread_score * 0.25 +
                         length_score * 0.30 +
                         area_score * 0.15)

        # 디버그 정보 저장
        debug_info = {
            'aspect': aspect_score,
            'spread': spread_score,
            'length': length_score,
            'area': area_score,
            'raw_conf': raw_confidence,
            'aspect_ratio': aspect_ratio,
            'spread_ratio': spread_ratio,
            'body_length': body_length_normalized
        }
        self.debug_scores[person_id] = debug_info

        # === 시간 필터링 (3프레임 연속 감지) ===
        is_fallen_raw = raw_confidence > 0.5

        if is_fallen_raw:
            self.fall_frame_counts[person_id] = self.fall_frame_counts.get(person_id, 0) + 1
        else:
            self.fall_frame_counts[person_id] = 0

        is_fallen = self.fall_frame_counts.get(person_id, 0) >= self.FALL_CONFIRM_FRAMES

        if is_fallen:
            confidence = raw_confidence
        else:
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
        
        # 1. 사람들 간의 거리 확인
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
            fighting_score += max(distance_scores) * 0.4
        
        # 2. 움직임 패턴 확인 (이전 프레임과 비교)
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
            fighting_score += max(movement_scores) * 0.3
        
        # 3. 쓰러짐 감지와 연계 (싸움 중 쓰러짐)
        fall_scores = [pose.fall_confidence for pose in current_poses if pose.is_fallen]
        if fall_scores and len(current_poses) >= 2:
            fighting_score += max(fall_scores) * 0.3
        
        is_fighting = fighting_score > 0.5
        return is_fighting, fighting_score
    
    def _remove_duplicate_poses(self, poses: List[PersonPose], 
                                frame_width: int, 
                                frame_height: int,
                                distance_threshold: float = 100.0) -> List[PersonPose]:
        """중복된 포즈 제거 (같은 사람이 여러 영역에서 감지된 경우)"""
        if len(poses) <= 1:
            return poses
        
        filtered_poses = []
        used = [False] * len(poses)
        
        for i, pose1 in enumerate(poses):
            if used[i]:
                continue
            
            # 같은 그룹의 포즈 찾기
            group = [pose1]
            used[i] = True
            
            for j, pose2 in enumerate(poses[i+1:], start=i+1):
                if used[j]:
                    continue
                
                dist = self.calculate_distance(
                    pose1.center,
                    pose2.center,
                    frame_width,
                    frame_height
                )
                
                if dist < distance_threshold:
                    group.append(pose2)
                    used[j] = True
            
            # 그룹에서 가장 신뢰도가 높은 포즈 선택
            if len(group) > 1:
                # 쓰러짐 감지가 있는 포즈 우선, 없으면 첫 번째
                fallen_poses = [p for p in group if p.is_fallen]
                if fallen_poses:
                    filtered_poses.append(max(fallen_poses, key=lambda p: p.fall_confidence))
                else:
                    filtered_poses.append(group[0])
            else:
                filtered_poses.append(group[0])
        
        return filtered_poses
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[PersonPose], bool, float]:
        """
        프레임 처리 및 이벤트 감지
        
        Args:
            frame: 입력 프레임 (BGR)
        
        Returns:
            (poses, is_fighting, fight_confidence): 포즈 리스트, 싸움 여부, 싸움 신뢰도
        """
        height, width = frame.shape[:2]
        current_poses: List[PersonPose] = []
        
        if self.enable_multi_person:
            # 다중 인물 감지: 프레임을 격자로 나누어 각 영역에서 포즈 감지
            cell_width = width // self.grid_cols
            cell_height = height // self.grid_rows
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # 각 셀 영역 추출
                    x_start = col * cell_width
                    y_start = row * cell_height
                    x_end = x_start + cell_width
                    y_end = y_start + cell_height
                    
                    # 마지막 셀은 나머지 영역 포함
                    if col == self.grid_cols - 1:
                        x_end = width
                    if row == self.grid_rows - 1:
                        y_end = height
                    
                    cell_frame = frame[y_start:y_end, x_start:x_end]
                    
                    if cell_frame.size == 0:
                        continue
                    
                    # RGB로 변환
                    rgb_cell = cv2.cvtColor(cell_frame, cv2.COLOR_BGR2RGB)
                    rgb_cell.flags.writeable = False
                    
                    # 포즈 추정
                    results = self.pose.process(rgb_cell)
                    
                    if results.pose_landmarks:
                        # 랜드마크 좌표를 전체 프레임 좌표로 변환
                        landmarks = results.pose_landmarks.landmark
                        xs = []
                        ys = []
                        normalized_coords = {}
                        
                        for idx, lm in enumerate(landmarks):
                            # 셀 내 상대 좌표를 전체 프레임 좌표로 변환
                            x_abs = (lm.x * cell_width + x_start) / width
                            y_abs = (lm.y * cell_height + y_start) / height
                            xs.append(x_abs)
                            ys.append(y_abs)
                            normalized_coords[idx] = (x_abs, y_abs)
                        
                        x_min, x_max = int(min(xs) * width), int(max(xs) * width)
                        y_min, y_max = int(min(ys) * height), int(max(ys) * height)
                        
                        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                        
                        # 중심점 계산
                        center_x = sum(xs) / len(xs)
                        center_y = sum(ys) / len(ys)
                        center = (center_x, center_y)
                        
                        # 쓰러짐 감지 (변환된 좌표 사용)
                        is_fallen, fall_confidence, _ = self.detect_fall(results.pose_landmarks, normalized_coords, person_id=len(current_poses))
                        
                        pose = PersonPose(
                            landmarks=results.pose_landmarks,  # 원본 랜드마크 (그리기용)
                            bbox=bbox,
                            center=center,
                            is_fallen=is_fallen,
                            fall_confidence=fall_confidence
                        )
                        current_poses.append(pose)
            
            # 중복 제거 (같은 사람이 여러 셀에서 감지된 경우)
            current_poses = self._remove_duplicate_poses(current_poses, width, height)
        
        else:
            # 단일 인물 감지 (기존 방식)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # 포즈 추정
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 바운딩 박스 계산
                landmarks = results.pose_landmarks.landmark
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                
                x_min, x_max = int(min(xs) * width), int(max(xs) * width)
                y_min, y_max = int(min(ys) * height), int(max(ys) * height)
                
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # 중심점 계산
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
                center = (center_x, center_y)
                
                # 쓰러짐 감지
                is_fallen, fall_confidence, _ = self.detect_fall(results.pose_landmarks, person_id=0)
                
                pose = PersonPose(
                    landmarks=results.pose_landmarks,
                    bbox=bbox,
                    center=center,
                    is_fallen=is_fallen,
                    fall_confidence=fall_confidence
                )
                current_poses.append(pose)
        
        # 싸움 감지
        is_fighting, fight_confidence = self.detect_fighting(current_poses, width, height)
        
        # 이전 포즈 업데이트
        self.previous_poses = current_poses
        
        return current_poses, is_fighting, fight_confidence
    
    def draw_pose(self, frame: np.ndarray, poses: List[PersonPose], 
                  is_fighting: bool, fight_confidence: float):
        """포즈 및 이벤트 시각화"""
        for pose in poses:
            # 포즈 랜드마크 그리기
            if pose.landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose.landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
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
        self.pose.close()

