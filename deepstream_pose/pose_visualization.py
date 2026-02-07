#!/usr/bin/env python3
"""
Human Pose Visualization Module
스켈레톤 및 키포인트 시각화를 위한 유틸리티

DeepStream nvdsosd를 사용하여 포즈를 화면에 그립니다.
"""

import pyds
from typing import List, Tuple, Optional


# 색상 정의 (RGBA)
COLORS = {
    'skeleton': (0.0, 1.0, 0.0, 1.0),      # 초록색 - 스켈레톤
    'keypoint': (1.0, 1.0, 0.0, 1.0),       # 노란색 - 키포인트
    'bbox_normal': (0.0, 1.0, 0.0, 1.0),   # 초록색 - 일반 박스
    'bbox_fallen': (1.0, 0.0, 0.0, 1.0),   # 빨간색 - 쓰러짐
    'bbox_fighting': (1.0, 0.5, 0.0, 1.0), # 주황색 - 싸움
    'text': (1.0, 1.0, 1.0, 1.0),          # 흰색 - 텍스트
    'alert_bg': (1.0, 0.0, 0.0, 0.7),      # 빨간 반투명 - 경고 배경
}

# 키포인트 인덱스
class Keypoint:
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
    NECK = 17

# 스켈레톤 연결 정의 및 색상
# (시작점, 끝점, 색상 RGB)
SKELETON_LINKS = [
    # 다리
    (Keypoint.LEFT_ANKLE, Keypoint.LEFT_KNEE, (0.0, 1.0, 0.0)),
    (Keypoint.LEFT_KNEE, Keypoint.LEFT_HIP, (0.0, 1.0, 0.0)),
    (Keypoint.RIGHT_ANKLE, Keypoint.RIGHT_KNEE, (0.0, 0.0, 1.0)),
    (Keypoint.RIGHT_KNEE, Keypoint.RIGHT_HIP, (0.0, 0.0, 1.0)),
    # 엉덩이
    (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP, (1.0, 1.0, 0.0)),
    # 몸통
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP, (0.0, 1.0, 0.0)),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP, (0.0, 0.0, 1.0)),
    (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER, (1.0, 1.0, 0.0)),
    # 팔
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW, (0.0, 1.0, 0.0)),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW, (0.0, 0.0, 1.0)),
    (Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST, (0.0, 1.0, 0.0)),
    (Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST, (0.0, 0.0, 1.0)),
    # 얼굴
    (Keypoint.LEFT_EYE, Keypoint.RIGHT_EYE, (1.0, 0.0, 1.0)),
    (Keypoint.NOSE, Keypoint.LEFT_EYE, (1.0, 0.0, 1.0)),
    (Keypoint.NOSE, Keypoint.RIGHT_EYE, (1.0, 0.0, 1.0)),
    (Keypoint.LEFT_EYE, Keypoint.LEFT_EAR, (0.0, 1.0, 0.0)),
    (Keypoint.RIGHT_EYE, Keypoint.RIGHT_EAR, (0.0, 0.0, 1.0)),
    # 목
    (Keypoint.NOSE, Keypoint.NECK, (1.0, 1.0, 0.0)),
    (Keypoint.NECK, Keypoint.LEFT_SHOULDER, (0.0, 1.0, 0.0)),
    (Keypoint.NECK, Keypoint.RIGHT_SHOULDER, (0.0, 0.0, 1.0)),
]

# 키포인트 색상 (파트별)
KEYPOINT_COLORS = {
    # 얼굴 (분홍)
    Keypoint.NOSE: (1.0, 0.5, 0.5),
    Keypoint.LEFT_EYE: (1.0, 0.5, 0.5),
    Keypoint.RIGHT_EYE: (1.0, 0.5, 0.5),
    Keypoint.LEFT_EAR: (1.0, 0.5, 0.5),
    Keypoint.RIGHT_EAR: (1.0, 0.5, 0.5),
    # 상체 왼쪽 (초록)
    Keypoint.LEFT_SHOULDER: (0.0, 1.0, 0.0),
    Keypoint.LEFT_ELBOW: (0.0, 1.0, 0.0),
    Keypoint.LEFT_WRIST: (0.0, 1.0, 0.0),
    # 상체 오른쪽 (파랑)
    Keypoint.RIGHT_SHOULDER: (0.0, 0.0, 1.0),
    Keypoint.RIGHT_ELBOW: (0.0, 0.0, 1.0),
    Keypoint.RIGHT_WRIST: (0.0, 0.0, 1.0),
    # 하체 왼쪽 (초록)
    Keypoint.LEFT_HIP: (0.0, 1.0, 0.0),
    Keypoint.LEFT_KNEE: (0.0, 1.0, 0.0),
    Keypoint.LEFT_ANKLE: (0.0, 1.0, 0.0),
    # 하체 오른쪽 (파랑)
    Keypoint.RIGHT_HIP: (0.0, 0.0, 1.0),
    Keypoint.RIGHT_KNEE: (0.0, 0.0, 1.0),
    Keypoint.RIGHT_ANKLE: (0.0, 0.0, 1.0),
    # 목 (노랑)
    Keypoint.NECK: (1.0, 1.0, 0.0),
}


class PoseVisualizer:
    """포즈 시각화 클래스"""

    def __init__(self, keypoint_radius: int = 5, line_width: int = 2):
        """
        Args:
            keypoint_radius: 키포인트 원 반지름
            line_width: 스켈레톤 선 두께
        """
        self.keypoint_radius = keypoint_radius
        self.line_width = line_width

    def draw_skeleton(
        self,
        display_meta: pyds.NvDsDisplayMeta,
        keypoints: List[Tuple[float, float, float]],
        frame_width: int,
        frame_height: int,
        offset_idx: int = 0
    ) -> int:
        """
        스켈레톤 그리기

        Args:
            display_meta: DeepStream 디스플레이 메타데이터
            keypoints: 키포인트 리스트 [(x, y, confidence), ...]
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            offset_idx: 시작 인덱스 (여러 사람 그리기 시)

        Returns:
            사용된 요소 수
        """
        line_idx = offset_idx

        for start_idx, end_idx, color in SKELETON_LINKS:
            if (start_idx >= len(keypoints) or end_idx >= len(keypoints)):
                continue

            kp1 = keypoints[start_idx]
            kp2 = keypoints[end_idx]

            # 신뢰도 확인
            if kp1[2] < 0.5 or kp2[2] < 0.5:
                continue

            if line_idx >= 16:  # NvDsDisplayMeta 제한
                break

            # 좌표 변환 (정규화 -> 픽셀)
            x1 = int(kp1[0] * frame_width)
            y1 = int(kp1[1] * frame_height)
            x2 = int(kp2[0] * frame_width)
            y2 = int(kp2[1] * frame_height)

            # 선 그리기
            line_params = display_meta.line_params[line_idx]
            line_params.x1 = x1
            line_params.y1 = y1
            line_params.x2 = x2
            line_params.y2 = y2
            line_params.line_width = self.line_width
            line_params.line_color.red = color[0]
            line_params.line_color.green = color[1]
            line_params.line_color.blue = color[2]
            line_params.line_color.alpha = 1.0

            line_idx += 1

        display_meta.num_lines = line_idx

        return line_idx - offset_idx

    def draw_keypoints(
        self,
        display_meta: pyds.NvDsDisplayMeta,
        keypoints: List[Tuple[float, float, float]],
        frame_width: int,
        frame_height: int,
        offset_idx: int = 0
    ) -> int:
        """
        키포인트 그리기

        Args:
            display_meta: DeepStream 디스플레이 메타데이터
            keypoints: 키포인트 리스트
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            offset_idx: 시작 인덱스

        Returns:
            사용된 요소 수
        """
        circle_idx = offset_idx

        for kp_idx, kp in enumerate(keypoints):
            if kp[2] < 0.5:  # 신뢰도 낮음
                continue

            if circle_idx >= 16:  # NvDsDisplayMeta 제한
                break

            x = int(kp[0] * frame_width)
            y = int(kp[1] * frame_height)

            # 색상
            color = KEYPOINT_COLORS.get(kp_idx, (1.0, 1.0, 0.0))

            # 원 그리기
            circle_params = display_meta.circle_params[circle_idx]
            circle_params.xc = x
            circle_params.yc = y
            circle_params.radius = self.keypoint_radius
            circle_params.circle_color.red = color[0]
            circle_params.circle_color.green = color[1]
            circle_params.circle_color.blue = color[2]
            circle_params.circle_color.alpha = 1.0
            circle_params.has_bg_color = 1
            circle_params.bg_color.red = color[0]
            circle_params.bg_color.green = color[1]
            circle_params.bg_color.blue = color[2]
            circle_params.bg_color.alpha = 1.0

            circle_idx += 1

        display_meta.num_circles = circle_idx

        return circle_idx - offset_idx

    def draw_bbox(
        self,
        display_meta: pyds.NvDsDisplayMeta,
        bbox: Tuple[int, int, int, int],
        is_fallen: bool = False,
        is_fighting: bool = False,
        person_id: int = 0,
        offset_idx: int = 0
    ) -> int:
        """
        바운딩 박스 그리기

        Args:
            display_meta: DeepStream 디스플레이 메타데이터
            bbox: (x, y, width, height)
            is_fallen: 쓰러짐 상태
            is_fighting: 싸움 상태
            person_id: 사람 ID
            offset_idx: 시작 인덱스

        Returns:
            사용된 요소 수
        """
        if offset_idx >= 16:
            return 0

        x, y, w, h = bbox

        # 색상 결정
        if is_fallen:
            color = COLORS['bbox_fallen']
        elif is_fighting:
            color = COLORS['bbox_fighting']
        else:
            color = COLORS['bbox_normal']

        # 박스 그리기 (4개의 선으로 구성)
        rect_params = display_meta.rect_params[offset_idx]
        rect_params.left = x
        rect_params.top = y
        rect_params.width = w
        rect_params.height = h
        rect_params.border_width = 2
        rect_params.border_color.red = color[0]
        rect_params.border_color.green = color[1]
        rect_params.border_color.blue = color[2]
        rect_params.border_color.alpha = color[3]
        rect_params.has_bg_color = 0

        display_meta.num_rects = offset_idx + 1

        return 1

    def draw_alert(
        self,
        display_meta: pyds.NvDsDisplayMeta,
        message: str,
        confidence: float,
        x: int = 10,
        y: int = 30
    ):
        """
        경고 메시지 그리기

        Args:
            display_meta: DeepStream 디스플레이 메타데이터
            message: 경고 메시지
            confidence: 신뢰도
            x: x 좌표
            y: y 좌표
        """
        if display_meta.num_labels >= 16:
            return

        txt_params = display_meta.text_params[display_meta.num_labels]
        txt_params.display_text = f"{message} ({confidence:.2f})"
        txt_params.x_offset = x
        txt_params.y_offset = y
        txt_params.font_params.font_name = "Arial"
        txt_params.font_params.font_size = 18
        txt_params.font_params.font_color.red = 1.0
        txt_params.font_params.font_color.green = 0.0
        txt_params.font_params.font_color.blue = 0.0
        txt_params.font_params.font_color.alpha = 1.0
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.red = 1.0
        txt_params.text_bg_clr.green = 1.0
        txt_params.text_bg_clr.blue = 1.0
        txt_params.text_bg_clr.alpha = 0.8

        display_meta.num_labels += 1

    def draw_stats(
        self,
        display_meta: pyds.NvDsDisplayMeta,
        fps: float,
        num_persons: int,
        fall_count: int,
        fight_count: int,
        frame_height: int
    ):
        """
        통계 정보 그리기

        Args:
            display_meta: DeepStream 디스플레이 메타데이터
            fps: 현재 FPS
            num_persons: 감지된 인물 수
            fall_count: 쓰러짐 감지 횟수
            fight_count: 싸움 감지 횟수
            frame_height: 프레임 높이
        """
        if display_meta.num_labels >= 16:
            return

        txt_params = display_meta.text_params[display_meta.num_labels]
        txt_params.display_text = (
            f"FPS: {fps:.1f} | Persons: {num_persons} | "
            f"Falls: {fall_count} | Fights: {fight_count}"
        )
        txt_params.x_offset = 10
        txt_params.y_offset = frame_height - 30
        txt_params.font_params.font_name = "Arial"
        txt_params.font_params.font_size = 12
        txt_params.font_params.font_color.red = 1.0
        txt_params.font_params.font_color.green = 1.0
        txt_params.font_params.font_color.blue = 1.0
        txt_params.font_params.font_color.alpha = 1.0
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.red = 0.0
        txt_params.text_bg_clr.green = 0.0
        txt_params.text_bg_clr.blue = 0.0
        txt_params.text_bg_clr.alpha = 0.7

        display_meta.num_labels += 1
