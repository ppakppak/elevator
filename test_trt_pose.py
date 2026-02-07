#!/usr/bin/env python3
"""
TensorRT YOLO11-Pose 테스트 스크립트
"""

import cv2
import time
import argparse
import os
from datetime import datetime
from pose_detector_trt import TRTPoseDetector


def main():
    parser = argparse.ArgumentParser(description='TensorRT Pose Detector Test')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index or file path)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display')
    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT YOLO11-Pose Test")
    print("=" * 60)

    # 감지기 초기화 (자동으로 .pt 또는 .engine 선택)
    detector = TRTPoseDetector(
        model_path='yolo11n-pose.pt',
        min_detection_confidence=0.5
    )

    # 비디오 소스 설정
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        return

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 저장 폴더 생성
    save_dir = "captured_frames"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Source: {args.source}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Press 's' to save frame, 'q' to quit")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 프레임 복사 (저장용)
        original_frame = frame.copy()

        # 포즈 감지
        t0 = time.time()
        poses, is_fighting, fight_confidence = detector.process_frame(frame)
        inference_time = (time.time() - t0) * 1000

        # 시각화
        detector.draw_pose(frame, poses, is_fighting, fight_confidence)

        # FPS 계산
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # 정보 표시
        info_text = f"FPS: {fps_display:.1f} | Inference: {inference_time:.1f}ms | Persons: {len(poses)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 쓰러짐 감지 표시
        fallen_count = sum(1 for p in poses if p.is_fallen)
        if fallen_count > 0:
            cv2.putText(frame, f"FALL DETECTED: {fallen_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if not args.no_display:
            cv2.imshow('TensorRT Pose Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 현재 프레임 저장 (원본 + 시각화)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # 원본 프레임 저장
                orig_filename = os.path.join(save_dir, f"orig_{timestamp}.jpg")
                cv2.imwrite(orig_filename, original_frame)
                # 시각화 프레임 저장
                vis_filename = os.path.join(save_dir, f"vis_{timestamp}.jpg")
                cv2.imwrite(vis_filename, frame)
                print(f"Saved: {orig_filename} (original), {vis_filename} (visualized)")

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    print("\nTest completed!")


if __name__ == "__main__":
    main()
