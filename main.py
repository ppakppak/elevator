"""
쓰러짐 및 싸움 감지 시스템
메인 실행 스크립트
MediaPipe 또는 YOLOv11-Pose 선택 가능
"""

import cv2
import argparse
import time
from pose_detector import PoseDetector
from web_streamer import WebStreamer


def main():
    parser = argparse.ArgumentParser(
        description='MediaPipe를 사용한 쓰러짐 및 싸움 감지 시스템'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='비디오 소스 (카메라 번호 또는 비디오 파일 경로, 기본값: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 비디오 파일 경로 (선택사항)'
    )
    parser.add_argument(
        '--min-detection-confidence',
        type=float,
        default=0.5,
        help='최소 감지 신뢰도 (기본값: 0.5)'
    )
    parser.add_argument(
        '--min-tracking-confidence',
        type=float,
        default=0.5,
        help='최소 추적 신뢰도 (기본값: 0.5)'
    )
    parser.add_argument(
        '--model-complexity',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='모델 복잡도 (0=가벼움, 1=균형, 2=무거움, 기본값: 1)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='화면 표시 비활성화 (SSH 환경용)'
    )
    parser.add_argument(
        '--web',
        action='store_true',
        help='웹 스트리밍 모드 활성화 (로컬 PC 브라우저에서 확인 가능)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='웹 서버 호스트 (기본값: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='웹 서버 포트 (기본값: 5000)'
    )
    parser.add_argument(
        '--resize-width',
        type=int,
        default=640,
        help='프레임 리사이즈 너비 (성능 최적화, 기본값: 640, 0이면 리사이즈 안함)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='프레임 스킵 (N개 중 1개만 처리, 기본값: 1=모든 프레임 처리)'
    )
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=70,
        help='JPEG 인코딩 품질 1-100 (낮을수록 빠름, 기본값: 70)'
    )
    parser.add_argument(
        '--no-drawing',
        action='store_true',
        help='포즈 그리기 비활성화 (성능 향상)'
    )
    parser.add_argument(
        '--multi-person',
        action='store_true',
        help='다중 인물 감지 활성화 (싸움 감지를 위해 필요)'
    )
    parser.add_argument(
        '--grid-rows',
        type=int,
        default=2,
        help='다중 인물 감지 격자 행 수 (기본값: 2)'
    )
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=2,
        help='다중 인물 감지 격자 열 수 (기본값: 2)'
    )
    parser.add_argument(
        '--use-yolo',
        action='store_true',
        help='YOLOv11-Pose 사용 (기본값: MediaPipe)'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv11-Pose 모델 크기 (n=빠름, x=정확, 기본값: n)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='사용할 디바이스 (auto=자동, cpu, cuda, 0 등, 기본값: auto)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화 (쓰러짐 감지 점수를 화면에 표시)'
    )
    parser.add_argument(
        '--ceiling',
        action='store_true',
        help='천장 카메라 모드 (앙상블 낙상 감지, YOLO와 함께 사용)'
    )
    parser.add_argument(
        '--use-seg',
        action='store_true',
        help='세그멘테이션 모델 사용 (학습된 모델로 직접 분류, 가장 정확)'
    )
    parser.add_argument(
        '--seg-model',
        type=str,
        default='m',
        choices=['m', 'l'],
        help='세그멘테이션 모델 크기 (m=빠름, l=정확, 기본값: m)'
    )

    args = parser.parse_args()
    
    # 천장 모드는 YOLO와 함께만 사용 가능
    if args.ceiling and not args.use_yolo and not args.use_seg:
        print("경고: 천장 카메라 모드는 YOLO와 함께 사용해야 합니다. --use-yolo 옵션을 추가합니다.")
        args.use_yolo = True

    # 세그멘테이션 모델 사용 (가장 정확)
    if args.use_seg:
        from detector_segmentation import create_detector
        detector = create_detector(
            model_size=args.seg_model,
            confidence=args.min_detection_confidence,
            device=args.device
        )
        print(f"세그멘테이션 모델 사용 (모델: best_{args.seg_model}.pt)")

        # 세그멘테이션 모델 전용 실행 루프
        try:
            source = int(args.source)
            cap = cv2.VideoCapture(source)
        except ValueError:
            cap = cv2.VideoCapture(args.source)

        if not cap.isOpened():
            print(f"오류: 비디오 소스를 열 수 없습니다: {args.source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        out = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        print("시작: 세그멘테이션 기반 쓰러짐/싸움 감지 시스템")
        print("종료: ESC 또는 'q' 키를 누르세요")
        print("-" * 50)

        # 디스플레이 윈도우 크기 제한 (세로 영상 대응)
        max_display_height = 720
        display_scale = 1.0
        if height > max_display_height:
            display_scale = max_display_height / height
            print(f"디스플레이 축소: {display_scale:.2f}x (원본 {width}x{height})")

        frame_count = 0
        start_time = time.time()
        fall_count = 0
        fight_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("비디오 종료 또는 프레임을 읽을 수 없습니다.")
                    break

                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 프레임 스킵
                if args.frame_skip > 1 and frame_num % args.frame_skip != 0:
                    continue

                # 감지
                detections = detector.process_frame(frame, frame_number=frame_num)

                # 이벤트 카운트
                for det in detections:
                    if det.is_fallen:
                        fall_count += 1
                        print(f"[쓰러짐 감지] 프레임 {frame_num}, 신뢰도: {det.confidence:.2f}")
                    if det.is_fighting:
                        fight_count += 1
                        print(f"[싸움 감지] 프레임 {frame_num}, 신뢰도: {det.confidence:.2f}")

                # 시각화
                if not args.no_drawing:
                    frame = detector.draw_results(frame, detections)

                # FPS 표시
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame,
                           f"FPS: {current_fps:.1f}",
                           (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (255, 255, 255),
                           2)

                cv2.putText(frame,
                           f"Falls: {fall_count} | Fights: {fight_count}",
                           (10, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (255, 255, 255),
                           2)

                if out:
                    out.write(frame)

                if not args.no_display:
                    # 디스플레이용 리사이즈 (저장은 원본 크기)
                    if display_scale < 1.0:
                        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
                    else:
                        display_frame = frame
                    cv2.imshow('Segmentation Detector', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
                else:
                    time.sleep(1.0 / fps)

        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")

        finally:
            cap.release()
            if out:
                out.release()
            if not args.no_display:
                cv2.destroyAllWindows()

            elapsed_time = time.time() - start_time
            print("-" * 50)
            print(f"처리 완료:")
            print(f"  총 프레임: {frame_count}")
            print(f"  처리 시간: {elapsed_time:.2f}초")
            print(f"  평균 FPS: {frame_count / elapsed_time:.2f}" if elapsed_time > 0 else "  평균 FPS: N/A")
            print(f"  쓰러짐 감지 횟수: {fall_count}")
            print(f"  싸움 감지 횟수: {fight_count}")
        return

    # 포즈 감지기 선택
    if args.use_yolo:
        try:
            from pose_detector_yolo import YOLOPoseDetector
            detector = YOLOPoseDetector(
                model_size=args.yolo_model,
                min_detection_confidence=args.min_detection_confidence,
                device=args.device,
                ceiling_mode=args.ceiling
            )
            mode_str = "천장 카메라 (앙상블)" if args.ceiling else "일반"
            print(f"YOLOv11-Pose 사용 (모델: yolov11{args.yolo_model}-pose.pt, 모드: {mode_str})")
        except ImportError:
            print("오류: ultralytics가 설치되지 않았습니다.")
            print("설치: pip install ultralytics")
            return
    else:
        detector = PoseDetector(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            model_complexity=args.model_complexity,
            enable_multi_person=args.multi_person,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols
        )
        print("MediaPipe Pose 사용")

    # 디버그 모드 설정
    if args.debug:
        detector.debug_mode = True
        print("디버그 모드 활성화 (쓰러짐 감지 점수 표시)")
        # 천장 모드일 때 앙상블 감지기 디버그도 활성화
        if args.ceiling and hasattr(detector, 'ceiling_detector') and detector.ceiling_detector:
            detector.ceiling_detector.debug_mode = True

    # 웹 스트리밍 모드
    if args.web:
        
        streamer = WebStreamer(
            detector=detector,
            source=args.source,
            host=args.host,
            port=args.port,
            resize_width=args.resize_width if args.resize_width > 0 else None,
            frame_skip=args.frame_skip,
            jpeg_quality=args.jpeg_quality,
            enable_drawing=not args.no_drawing
        )
        streamer.start()
        return
    
    # 비디오 소스 설정
    try:
        source = int(args.source)
        cap = cv2.VideoCapture(source)
    except ValueError:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"오류: 비디오 소스를 열 수 없습니다: {args.source}")
        return
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # 출력 비디오 설정
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 포즈 감지기 선택 (웹 모드가 아닌 경우)
    if args.use_yolo:
        try:
            from pose_detector_yolo import YOLOPoseDetector
            detector = YOLOPoseDetector(
                model_size=args.yolo_model,
                min_detection_confidence=args.min_detection_confidence,
                device=args.device,
                ceiling_mode=args.ceiling
            )
            mode_str = "천장 카메라 (앙상블)" if args.ceiling else "일반"
            print(f"YOLOv11-Pose 사용 (모델: yolov11{args.yolo_model}-pose.pt, 모드: {mode_str})")
        except ImportError:
            print("오류: ultralytics가 설치되지 않았습니다.")
            print("설치: pip install ultralytics")
            return
    else:
        detector = PoseDetector(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            model_complexity=args.model_complexity,
            enable_multi_person=args.multi_person,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols
        )
        print("MediaPipe Pose 사용")

    # 디버그 모드 설정 (비웹 모드)
    if args.debug:
        detector.debug_mode = True
        print("디버그 모드 활성화 (쓰러짐 감지 점수 표시)")
        # 천장 모드일 때 앙상블 감지기 디버그도 활성화
        if args.ceiling and hasattr(detector, 'ceiling_detector') and detector.ceiling_detector:
            detector.ceiling_detector.debug_mode = True

    print("시작: 쓰러짐 및 싸움 감지 시스템")
    print("종료: ESC 또는 'q' 키를 누르세요")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    fall_count = 0
    fight_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("비디오 종료 또는 프레임을 읽을 수 없습니다.")
                break

            # 프레임 카운터 (영상의 실제 프레임 번호)
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 프레임 처리 (실제 프레임 번호 전달)
            poses, is_fighting, fight_confidence = detector.process_frame(frame, frame_number=frame_count)
            
            # 이벤트 카운트
            for pose in poses:
                if pose.is_fallen:
                    fall_count += 1
                    print(f"[쓰러짐 감지] 프레임 {frame_count}, 신뢰도: {pose.fall_confidence:.2f}")
            
            if is_fighting:
                fight_count += 1
                print(f"[싸움 감지] 프레임 {frame_count}, 신뢰도: {fight_confidence:.2f}")
            
            # 시각화
            detector.draw_pose(frame, poses, is_fighting, fight_confidence)
            
            # FPS 표시
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame,
                       f"FPS: {current_fps:.1f}",
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (255, 255, 255),
                       2)
            
            # 통계 표시
            cv2.putText(frame,
                       f"Falls: {fall_count} | Fights: {fight_count}",
                       (10, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (255, 255, 255),
                       2)
            
            # 출력 비디오에 저장
            if out:
                out.write(frame)
            
            # 화면 표시
            if not args.no_display:
                cv2.imshow('Fall & Fight Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC 또는 'q'
                    break
            else:
                # SSH 환경에서는 프레임 처리만 수행
                time.sleep(1.0 / fps)
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    
    finally:
        # 리소스 해제
        detector.release()
        cap.release()
        if out:
            out.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # 최종 통계
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print(f"처리 완료:")
        print(f"  총 프레임: {frame_count}")
        print(f"  처리 시간: {elapsed_time:.2f}초")
        print(f"  평균 FPS: {frame_count / elapsed_time:.2f}" if elapsed_time > 0 else "  평균 FPS: N/A")
        print(f"  쓰러짐 감지 횟수: {fall_count}")
        print(f"  싸움 감지 횟수: {fight_count}")


if __name__ == '__main__':
    main()

