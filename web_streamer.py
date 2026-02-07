"""
Flask 기반 웹 스트리밍 서버
SSH 환경에서 실행하면서 로컬 PC 브라우저로 실시간 확인 가능
"""

import cv2
import threading
import time
import socket
from flask import Flask, Response, render_template_string, jsonify, request


class WebStreamer:
    """웹 스트리밍 서버 클래스"""
    
    def __init__(self, 
                 detector,
                 source=0,
                 host='0.0.0.0',
                 port=5000,
                 fps=30,
                 resize_width=640,
                 resize_height=None,
                 frame_skip=1,
                 jpeg_quality=70,
                 enable_drawing=True):
        """
        Args:
            detector: PoseDetector 또는 YOLOPoseDetector 인스턴스
            source: 비디오 소스 (카메라 번호 또는 파일 경로)
            host: 서버 호스트 (기본값: 0.0.0.0 - 모든 인터페이스)
            port: 서버 포트 (기본값: 5000)
            fps: 목표 FPS
            resize_width: 리사이즈 너비 (성능 최적화, None이면 리사이즈 안함)
            resize_height: 리사이즈 높이 (None이면 비율 유지)
            frame_skip: 프레임 스킵 (N개 중 1개만 처리, 기본값: 1 = 모든 프레임 처리)
            jpeg_quality: JPEG 인코딩 품질 (1-100, 낮을수록 빠름, 기본값: 70)
            enable_drawing: 포즈 그리기 활성화 (False면 성능 향상)
        """
        self.detector = detector
        self.source = source
        self.host = host
        self.port = self._find_available_port(port)
        self.fps = fps
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.frame_skip = max(1, frame_skip)  # 최소 1
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self.enable_drawing = enable_drawing
        
        self.app = Flask(__name__)
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.is_video_file = False
        
        self.frame_count = 0
        self.processed_frame_count = 0
        self.start_time = time.time()
        self.fall_count = 0
        self.fight_count = 0

        # 비디오 탐색 관련
        self.total_frames = 0
        self.video_fps = 30
        self.current_position = 0  # 현재 프레임 위치
        self.seek_position = None  # 탐색 요청 위치
        self.paused = False  # 일시정지 상태

        # 출력 비디오 저장 관련
        self.output_video = None
        self.output_path = None
        self.save_video = False

        # Flask 라우트 설정
        self.setup_routes()
    
    def _find_available_port(self, start_port):
        """사용 가능한 포트 찾기"""
        port = start_port
        max_attempts = 10
        
        for _ in range(max_attempts):
            if self._is_port_available(port):
                if port != start_port:
                    print(f"포트 {start_port}가 사용 중입니다. 포트 {port}를 사용합니다.")
                return port
            port += 1
        
        raise RuntimeError(f"사용 가능한 포트를 찾을 수 없습니다 ({start_port}-{port-1})")
    
    def _is_port_available(self, port):
        """포트 사용 가능 여부 확인"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, port))
                return True
            except OSError as e:
                return False
    
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 페이지"""
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/video_feed')
        def video_feed():
            """비디오 스트리밍 엔드포인트"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            """통계 정보 JSON API"""
            elapsed_time = time.time() - self.start_time
            processed_fps = self.processed_frame_count / elapsed_time if elapsed_time > 0 else 0
            total_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return jsonify({
                'frame_count': self.frame_count,
                'processed_frame_count': self.processed_frame_count,
                'processed_fps': round(processed_fps, 2),
                'total_fps': round(total_fps, 2),
                'fall_count': self.fall_count,
                'fight_count': self.fight_count,
                'elapsed_time': round(elapsed_time, 2),
                # 비디오 탐색 정보
                'is_video_file': self.is_video_file,
                'total_frames': self.total_frames,
                'current_position': self.current_position,
                'video_fps': self.video_fps,
                'duration': round(self.total_frames / self.video_fps, 1) if self.video_fps > 0 else 0,
                'current_time': round(self.current_position / self.video_fps, 1) if self.video_fps > 0 else 0,
                'paused': self.paused,
                # 비디오 저장 상태
                'save_video': self.save_video,
                'output_path': self.output_path
            })

        @self.app.route('/seek', methods=['POST'])
        def seek():
            """비디오 탐색 API"""
            if not self.is_video_file:
                return jsonify({'error': '비디오 파일이 아닙니다'}), 400

            data = request.get_json()
            if 'position' in data:
                # 프레임 위치로 탐색
                position = int(data['position'])
                position = max(0, min(position, self.total_frames - 1))
                self.seek_position = position
                return jsonify({'success': True, 'position': position})
            elif 'time' in data:
                # 시간(초)으로 탐색
                time_sec = float(data['time'])
                position = int(time_sec * self.video_fps)
                position = max(0, min(position, self.total_frames - 1))
                self.seek_position = position
                return jsonify({'success': True, 'position': position, 'time': time_sec})
            elif 'delta' in data:
                # 상대적 탐색 (초 단위)
                delta_sec = float(data['delta'])
                delta_frames = int(delta_sec * self.video_fps)
                position = self.current_position + delta_frames
                position = max(0, min(position, self.total_frames - 1))
                self.seek_position = position
                return jsonify({'success': True, 'position': position})

            return jsonify({'error': '잘못된 요청'}), 400

        @self.app.route('/pause', methods=['POST'])
        def pause():
            """일시정지/재생 토글 API"""
            self.paused = not self.paused
            return jsonify({'paused': self.paused})

        @self.app.route('/config', methods=['GET'])
        def get_config():
            """현재 설정 조회 API"""
            config = {
                'enable_drawing': self.enable_drawing,
                'frame_skip': self.frame_skip,
                'jpeg_quality': self.jpeg_quality,
                'resize_width': self.resize_width,
                'debug_mode': getattr(self.detector, 'debug_mode', False),
            }
            
            # 감지기별 설정 추가
            if hasattr(self.detector, 'min_confidence'):
                config['detection_confidence'] = self.detector.min_confidence
            elif hasattr(self.detector, 'min_detection_confidence'):
                config['detection_confidence'] = self.detector.min_detection_confidence
            
            # 세그멘테이션 모델인 경우
            if hasattr(self.detector, 'confidence'):
                config['detection_confidence'] = self.detector.confidence
            
            # 출력 비디오 상태
            config['save_video'] = self.save_video
            config['output_path'] = self.output_path if self.save_video else None
            
            return jsonify(config)

        @self.app.route('/config', methods=['POST'])
        def update_config():
            """설정 변경 API"""
            data = request.get_json()
            changes = {}
            
            # 포즈 그리기 on/off
            if 'enable_drawing' in data:
                self.enable_drawing = bool(data['enable_drawing'])
                changes['enable_drawing'] = self.enable_drawing
            
            # 디버그 모드 on/off
            if 'debug_mode' in data:
                if hasattr(self.detector, 'debug_mode'):
                    self.detector.debug_mode = bool(data['debug_mode'])
                    changes['debug_mode'] = self.detector.debug_mode
                    # 천장 모드일 때 앙상블 감지기 디버그도 활성화
                    if hasattr(self.detector, 'ceiling_detector') and self.detector.ceiling_detector:
                        self.detector.ceiling_detector.debug_mode = self.detector.debug_mode
            
            # 감지 신뢰도 조정
            if 'detection_confidence' in data:
                conf = float(data['detection_confidence'])
                conf = max(0.0, min(1.0, conf))  # 0.0~1.0 범위로 제한
                
                if hasattr(self.detector, 'min_confidence'):
                    self.detector.min_confidence = conf
                    changes['detection_confidence'] = conf
                elif hasattr(self.detector, 'min_detection_confidence'):
                    self.detector.min_detection_confidence = conf
                    changes['detection_confidence'] = conf
                elif hasattr(self.detector, 'confidence'):
                    self.detector.confidence = conf
                    changes['detection_confidence'] = conf
            
            # 프레임 스킵 조정
            if 'frame_skip' in data:
                skip = int(data['frame_skip'])
                skip = max(1, skip)  # 최소 1
                self.frame_skip = skip
                changes['frame_skip'] = skip
            
            # JPEG 품질 조정
            if 'jpeg_quality' in data:
                quality = int(data['jpeg_quality'])
                quality = max(1, min(100, quality))  # 1~100 범위
                self.jpeg_quality = quality
                changes['jpeg_quality'] = quality
            
            # 리사이즈 너비 조정
            if 'resize_width' in data:
                width = int(data['resize_width'])
                if width > 0:
                    self.resize_width = width
                    changes['resize_width'] = width
                elif width == 0:
                    self.resize_width = None
                    changes['resize_width'] = None
            
            return jsonify({'success': True, 'changes': changes})

        @self.app.route('/video_output', methods=['POST'])
        def control_video_output():
            """출력 비디오 저장 제어 API"""
            data = request.get_json()
            
            if 'action' not in data:
                return jsonify({'error': 'action 필드가 필요합니다'}), 400
            
            action = data['action']
            
            if action == 'start':
                # 출력 비디오 저장 시작
                if self.save_video:
                    return jsonify({'error': '이미 저장 중입니다'}), 400
                
                output_path = data.get('output_path', f'output_{int(time.time())}.mp4')
                self.output_path = output_path
                self.save_video = True
                
                # VideoWriter 초기화는 process_video 스레드에서 수행
                return jsonify({
                    'success': True,
                    'message': f'비디오 저장 시작: {output_path}',
                    'output_path': output_path
                })
            
            elif action == 'stop':
                # 출력 비디오 저장 중지
                if not self.save_video:
                    return jsonify({'error': '저장 중이 아닙니다'}), 400
                
                self.save_video = False
                if self.output_video:
                    self.output_video.release()
                    self.output_video = None
                
                saved_path = self.output_path
                self.output_path = None
                
                return jsonify({
                    'success': True,
                    'message': f'비디오 저장 완료: {saved_path}',
                    'output_path': saved_path
                })
            
            else:
                return jsonify({'error': '잘못된 action입니다 (start/stop)'}), 400

        @self.app.route('/reset_stats', methods=['POST'])
        def reset_stats():
            """통계 리셋 API"""
            self.fall_count = 0
            self.fight_count = 0
            self.frame_count = 0
            self.processed_frame_count = 0
            self.start_time = time.time()
            return jsonify({'success': True, 'message': '통계가 리셋되었습니다'})
    
    def generate_frames(self):
        """프레임 생성 제너레이터 (프레임 캐싱으로 번쩍임 방지)"""
        last_frame_bytes = None
        while self.running:
            current_bytes = None
            with self.lock:
                if self.frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ret:
                        current_bytes = buffer.tobytes()
                        last_frame_bytes = current_bytes
            
            # 현재 프레임 없으면 마지막 프레임 재사용
            if current_bytes is None and last_frame_bytes is not None:
                current_bytes = last_frame_bytes
            
            if current_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
    
    def process_video(self):
        """비디오 처리 스레드"""
        try:
            # 비디오 소스 열기
            try:
                source = int(self.source)
                self.cap = cv2.VideoCapture(source)
                self.is_video_file = False
            except ValueError:
                self.cap = cv2.VideoCapture(self.source)
                self.is_video_file = True
            
            if not self.cap.isOpened():
                print(f"오류: 비디오 소스를 열 수 없습니다: {self.source}")
                return
            
            # 비디오 속성 가져오기
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.video_fps = video_fps

            # 비디오 파일인 경우 총 프레임 수 저장
            if self.is_video_file:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = self.total_frames / video_fps if video_fps > 0 else 0
                print(f"비디오 길이: {duration:.1f}초 ({self.total_frames} 프레임)")

            # 리사이즈 계산
            if self.resize_width and original_width > self.resize_width:
                scale = self.resize_width / original_width
                width = self.resize_width
                height = int(original_height * scale) if not self.resize_height else self.resize_height
            else:
                width = original_width
                height = original_height
            
            print(f"원본 해상도: {original_width}x{original_height}")
            print(f"처리 해상도: {width}x{height}")
            print(f"비디오 FPS: {video_fps:.2f}")
            print(f"프레임 스킵: {self.frame_skip} (N개 중 1개 처리)")
            print(f"JPEG 품질: {self.jpeg_quality}")
            print(f"포즈 그리기: {'활성화' if self.enable_drawing else '비활성화'}")
            print(f"웹 스트리밍 서버 시작: http://{self.get_server_ip()}:{self.port}")
            print("-" * 50)
            
            frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / self.fps
            skip_counter = 0

            while self.running:
                # 일시정지 처리
                if self.paused:
                    time.sleep(0.1)
                    # 탐색 요청이 있으면 처리
                    if self.seek_position is not None:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_position)
                        self.current_position = self.seek_position
                        self.seek_position = None
                        # 탐색 후 현재 프레임 표시
                        ret, frame = self.cap.read()
                        if ret:
                            if self.resize_width and original_width > self.resize_width:
                                frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            else:
                                frame_resized = frame
                            # 포즈 처리 (실제 프레임 번호 전달)
                            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                            poses, is_fighting, fight_confidence = self.detector.process_frame(frame_resized, frame_number=frame_num)
                            if self.enable_drawing:
                                self.detector.draw_pose(frame_resized, poses, is_fighting, fight_confidence)
                            with self.lock:
                                self.frame = frame_resized.copy()
                            # 프레임 위치 되돌리기 (일시정지 상태에서 같은 프레임 유지)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_position)
                    continue

                # 탐색 요청 처리
                if self.seek_position is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_position)
                    self.current_position = self.seek_position
                    self.seek_position = None
                    skip_counter = 0

                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_file:
                        # 비디오 파일의 경우 처음부터 다시 재생 (프레임 유지)
                        print("비디오 파일 재생 완료. 처음부터 다시 재생합니다...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_position = 0
                        skip_counter = 0
                        time.sleep(0.1)  # 부드러운 전환을 위한 딜레이
                        continue
                    else:
                        print("비디오 종료 또는 프레임을 읽을 수 없습니다.")
                        break

                # 현재 위치 업데이트
                self.current_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.frame_count += 1
                
                # 프레임 스킵 처리
                skip_counter += 1
                if skip_counter < self.frame_skip:
                    # 스킵된 프레임은 리사이즈만 하고 스트리밍용으로 저장
                    if self.resize_width and original_width > self.resize_width:
                        frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_resized = frame.copy()
                    
                    with self.lock:
                        self.frame = frame_resized
                    continue
                
                skip_counter = 0
                self.processed_frame_count += 1
                
                # 프레임 리사이즈 (성능 최적화) - 처리 전에 리사이즈
                if self.resize_width and original_width > self.resize_width:
                    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_resized = frame
                
                # 프레임 처리 (리사이즈된 프레임 사용 - 성능 향상)
                frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                poses, is_fighting, fight_confidence = self.detector.process_frame(frame_resized, frame_number=frame_num)
                
                # 시각화는 리사이즈된 프레임에 직접 수행 (성능 향상)
                frame = frame_resized
                
                # 이벤트 카운트
                for pose in poses:
                    if pose.is_fallen:
                        self.fall_count += 1
                        print(f"[쓰러짐 감지] 프레임 {self.frame_count}, 신뢰도: {pose.fall_confidence:.2f}")
                
                if is_fighting:
                    self.fight_count += 1
                    print(f"[싸움 감지] 프레임 {self.frame_count}, 신뢰도: {fight_confidence:.2f}")
                
                # 시각화 (옵션)
                if self.enable_drawing:
                    self.detector.draw_pose(frame, poses, is_fighting, fight_confidence)
                
                # FPS 표시
                elapsed_time = time.time() - self.start_time
                processed_fps = self.processed_frame_count / elapsed_time if elapsed_time > 0 else 0
                total_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(frame,
                           f"FPS: {processed_fps:.1f} (Total: {total_fps:.1f})",
                           (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 255, 255),
                           2)
                
                # 통계 표시
                cv2.putText(frame,
                           f"Falls: {self.fall_count} | Fights: {self.fight_count}",
                           (10, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 255, 255),
                           2)
                
                # 출력 비디오 저장 (일시정지 상태가 아닐 때만)
                if self.save_video and not self.paused:
                    if self.output_video is None:
                        # VideoWriter 초기화
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.output_video = cv2.VideoWriter(
                            self.output_path, fourcc, video_fps, (width, height)
                        )
                        print(f"비디오 저장 시작: {self.output_path}")
                    self.output_video.write(frame)
                
                # 프레임 업데이트 (스레드 안전)
                with self.lock:
                    self.frame = frame.copy()
                
                # FPS 제어 (비디오 파일의 경우 원본 FPS 사용)
                time.sleep(frame_delay)
        
        except Exception as e:
            print(f"비디오 처리 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
            if self.output_video:
                self.output_video.release()
                self.output_video = None
    
    def get_server_ip(self):
        """서버 IP 주소 가져오기"""
        try:
            # 외부 연결을 위한 IP 주소 가져오기
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    def start(self):
        """서버 시작"""
        self.running = True
        
        # 포트 사용 가능 여부 확인
        if not self._is_port_available(self.port):
            print(f"\n⚠️  경고: 포트 {self.port}가 이미 사용 중입니다!")
            print(f"다른 포트를 사용하거나 기존 프로세스를 종료하세요.")
            print(f"포트 확인: lsof -i :{self.port} 또는 netstat -tulpn | grep {self.port}")
            return
        
        # 서버 정보 출력
        server_ip = self.get_server_ip()
        print(f"\n{'='*50}")
        print("웹 스트리밍 서버 시작 중...")
        print(f"{'='*50}")
        print(f"호스트: {self.host}")
        print(f"포트: {self.port}")
        print(f"서버 IP: {server_ip}")
        print(f"\n브라우저에서 다음 주소로 접속하세요:")
        print(f"  🌐 http://{server_ip}:{self.port}")
        if server_ip != "localhost":
            print(f"  🖥️  http://localhost:{self.port} (같은 머신에서)")
        print(f"{'='*50}\n")
        
        # 비디오 처리 스레드 시작 (Flask 서버 시작 전에 시작)
        video_thread = threading.Thread(target=self.process_video, daemon=True)
        video_thread.start()
        
        # 비디오 스레드가 초기화될 때까지 잠시 대기
        time.sleep(0.5)
        
        # Flask 서버 시작
        print("✅ 서버가 시작되었습니다!")
        print("서버를 중지하려면 Ctrl+C를 누르세요.\n")
        
        try:
            self.app.run(
                host=self.host, 
                port=self.port, 
                debug=False, 
                threaded=True, 
                use_reloader=False
            )
        except OSError as e:
            print(f"\n❌ 서버 시작 오류: {e}")
            if "Address already in use" in str(e) or "98" in str(e):
                print(f"\n포트 {self.port}가 이미 사용 중입니다.")
                print(f"해결 방법:")
                print(f"  1. 다른 포트 사용: --port 8080")
                print(f"  2. 기존 프로세스 종료:")
                print(f"     lsof -ti:{self.port} | xargs kill -9")
                print(f"     또는")
                print(f"     sudo fuser -k {self.port}/tcp")
            elif "Permission denied" in str(e) or "13" in str(e):
                print(f"\n포트 {self.port}에 접근할 권한이 없습니다.")
                print(f"해결 방법:")
                print(f"  1. 1024 이상의 포트 사용: --port 8080")
                print(f"  2. 또는 sudo로 실행 (권장하지 않음)")
            else:
                print(f"알 수 없는 오류입니다. 네트워크 설정을 확인하세요.")
            self.stop()
        except KeyboardInterrupt:
            print("\n서버를 종료합니다...")
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
        finally:
            self.stop()
    
    def stop(self):
        """서버 중지"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.detector.release()
        
        # 최종 통계
        elapsed_time = time.time() - self.start_time
        processed_fps = self.processed_frame_count / elapsed_time if elapsed_time > 0 else 0
        total_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("-" * 50)
        print(f"처리 완료:")
        print(f"  총 프레임: {self.frame_count}")
        print(f"  처리된 프레임: {self.processed_frame_count}")
        print(f"  처리 시간: {elapsed_time:.2f}초")
        print(f"  처리 FPS: {processed_fps:.2f}" if elapsed_time > 0 else "  처리 FPS: N/A")
        print(f"  전체 FPS: {total_fps:.2f}" if elapsed_time > 0 else "  전체 FPS: N/A")
        print(f"  쓰러짐 감지 횟수: {self.fall_count}")
        print(f"  싸움 감지 횟수: {self.fight_count}")


# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>쓰러짐 및 싸움 감지 시스템</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 30px;
        }
        .video-container {
            text-align: center;
            margin-bottom: 30px;
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 1.1em;
            color: #cccccc;
        }
        .alert {
            background-color: #ff4444;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            display: none;
        }
        .alert.show {
            display: block;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        /* 비디오 컨트롤 스타일 */
        .video-controls {
            background-color: #2a2a2a;
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }
        .video-controls.show {
            display: block;
        }
        .progress-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        .progress-bar {
            flex: 1;
            height: 8px;
            background-color: #444;
            border-radius: 4px;
            cursor: pointer;
            position: relative;
        }
        .progress-bar:hover {
            height: 12px;
        }
        .progress-fill {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 4px;
            width: 0%;
            transition: width 0.1s;
        }
        .time-display {
            font-family: monospace;
            font-size: 14px;
            color: #ccc;
            min-width: 120px;
            text-align: center;
        }
        .control-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .control-btn {
            background-color: #444;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .control-btn:hover {
            background-color: #555;
        }
        .control-btn.active {
            background-color: #4CAF50;
        }
        .control-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        /* 제어 패널 스타일 */
        .control-panel {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .control-panel h2 {
            color: #4CAF50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        .control-group {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #444;
        }
        .control-group:last-child {
            border-bottom: none;
        }
        .control-group label {
            display: block;
            margin-bottom: 8px;
            color: #ccc;
            font-size: 0.95em;
        }
        .control-group input[type="range"] {
            width: 100%;
            margin-bottom: 5px;
        }
        .control-group input[type="text"],
        .control-group input[type="number"] {
            width: 100%;
            padding: 8px;
            background-color: #333;
            border: 1px solid #555;
            border-radius: 4px;
            color: white;
            font-size: 14px;
        }
        .control-group .value-display {
            display: inline-block;
            margin-left: 10px;
            color: #4CAF50;
            font-weight: bold;
            min-width: 50px;
        }
        .control-group .checkbox-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .control-group .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .control-group button {
            flex: 1;
            min-width: 120px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-indicator.active {
            background-color: #4CAF50;
            box-shadow: 0 0 8px #4CAF50;
        }
        .status-indicator.inactive {
            background-color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 쓰러짐 및 싸움 감지 시스템</h1>
        
        <div id="alert" class="alert">
            ⚠️ 이상 행동 감지됨!
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Feed">

            <!-- 비디오 컨트롤 (비디오 파일일 때만 표시) -->
            <div id="videoControls" class="video-controls">
                <div class="progress-container">
                    <span class="time-display" id="timeDisplay">00:00 / 00:00</span>
                    <div class="progress-bar" id="progressBar" onclick="seekToPosition(event)">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>
                <div class="control-buttons">
                    <button class="control-btn" onclick="seek(-10)">⏪ -10초</button>
                    <button class="control-btn" onclick="seek(-5)">◀ -5초</button>
                    <button class="control-btn" onclick="seek(-1)">◁ -1초</button>
                    <button class="control-btn active" id="pauseBtn" onclick="togglePause()">⏸ 일시정지</button>
                    <button class="control-btn" onclick="seek(1)">▷ +1초</button>
                    <button class="control-btn" onclick="seek(5)">▶ +5초</button>
                    <button class="control-btn" onclick="seek(10)">⏩ +10초</button>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">쓰러짐 감지</div>
                <div class="stat-value" id="falls">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">싸움 감지</div>
                <div class="stat-value" id="fights">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">처리 프레임</div>
                <div class="stat-value" id="frames">0</div>
            </div>
        </div>

        <!-- 제어 패널 -->
        <div class="control-panel">
            <h2>⚙️ 프로그램 제어</h2>
            
            <!-- 감지 설정 -->
            <div class="control-group">
                <h3 style="color: #4CAF50; margin-top: 0; margin-bottom: 15px;">감지 설정</h3>
                <label>
                    감지 신뢰도: <span class="value-display" id="confidenceValue">0.5</span>
                    <input type="range" id="confidenceSlider" min="0" max="1" step="0.01" value="0.5" 
                           oninput="updateConfidence(this.value)">
                </label>
                <div class="checkbox-container" style="margin-top: 15px;">
                    <input type="checkbox" id="debugMode" onchange="updateDebugMode(this.checked)">
                    <label for="debugMode">디버그 모드 (쓰러짐 감지 점수 표시)</label>
                    <span class="status-indicator inactive" id="debugIndicator"></span>
                </div>
                <div class="checkbox-container" style="margin-top: 10px;">
                    <input type="checkbox" id="enableDrawing" checked onchange="updateDrawing(this.checked)">
                    <label for="enableDrawing">포즈 그리기 활성화</label>
                    <span class="status-indicator active" id="drawingIndicator"></span>
                </div>
            </div>

            <!-- 성능 설정 -->
            <div class="control-group">
                <h3 style="color: #4CAF50; margin-top: 0; margin-bottom: 15px;">성능 설정</h3>
                <label>
                    프레임 스킵: <span class="value-display" id="frameSkipValue">1</span> (N개 중 1개 처리)
                    <input type="range" id="frameSkipSlider" min="1" max="10" step="1" value="1" 
                           oninput="updateFrameSkip(this.value)">
                </label>
                <label style="margin-top: 15px;">
                    JPEG 품질: <span class="value-display" id="jpegQualityValue">70</span>
                    <input type="range" id="jpegQualitySlider" min="10" max="100" step="5" value="70" 
                           oninput="updateJpegQuality(this.value)">
                </label>
                <label style="margin-top: 15px;">
                    리사이즈 너비: <span class="value-display" id="resizeWidthValue">640</span>px (0=원본)
                    <input type="range" id="resizeWidthSlider" min="0" max="1920" step="80" value="640" 
                           oninput="updateResizeWidth(this.value)">
                </label>
            </div>

            <!-- 비디오 저장 -->
            <div class="control-group">
                <h3 style="color: #4CAF50; margin-top: 0; margin-bottom: 15px;">비디오 저장</h3>
                <label>
                    출력 파일 경로:
                    <input type="text" id="outputPath" value="output_video.mp4" placeholder="output_video.mp4">
                </label>
                <div class="button-group" style="margin-top: 15px;">
                    <button class="control-btn" id="startRecordBtn" onclick="startVideoOutput()">
                        🎥 저장 시작
                    </button>
                    <button class="control-btn" id="stopRecordBtn" onclick="stopVideoOutput()" disabled>
                        ⏹ 저장 중지
                    </button>
                </div>
                <div id="recordStatus" style="margin-top: 10px; color: #ccc; font-size: 0.9em;"></div>
            </div>

            <!-- 통계 제어 -->
            <div class="control-group">
                <h3 style="color: #4CAF50; margin-top: 0; margin-bottom: 15px;">통계 제어</h3>
                <div class="button-group">
                    <button class="control-btn" onclick="resetStats()">🔄 통계 리셋</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isPaused = false;
        let isVideoFile = false;
        let totalDuration = 0;

        // 시간 포맷 (초 -> MM:SS)
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // 통계 업데이트
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.processed_fps.toFixed(1);
                    document.getElementById('falls').textContent = data.fall_count;
                    document.getElementById('fights').textContent = data.fight_count;
                    document.getElementById('frames').textContent = data.frame_count;

                    // 경고 표시
                    const alert = document.getElementById('alert');
                    if (data.fall_count > 0 || data.fight_count > 0) {
                        alert.classList.add('show');
                    } else {
                        alert.classList.remove('show');
                    }

                    // 비디오 컨트롤 표시/업데이트
                    isVideoFile = data.is_video_file;
                    if (isVideoFile) {
                        document.getElementById('videoControls').classList.add('show');
                        totalDuration = data.duration;

                        // 프로그레스 바 업데이트
                        const progress = (data.current_time / data.duration) * 100;
                        document.getElementById('progressFill').style.width = progress + '%';
                        document.getElementById('timeDisplay').textContent =
                            formatTime(data.current_time) + ' / ' + formatTime(data.duration);

                        // 일시정지 버튼 상태
                        isPaused = data.paused;
                        const pauseBtn = document.getElementById('pauseBtn');
                        if (isPaused) {
                            pauseBtn.textContent = '▶ 재생';
                            pauseBtn.classList.add('active');
                        } else {
                            pauseBtn.textContent = '⏸ 일시정지';
                            pauseBtn.classList.remove('active');
                        }
                    }
                })
                .catch(error => console.error('통계 업데이트 오류:', error));
        }

        // 탐색 (초 단위 델타)
        function seek(deltaSec) {
            fetch('/seek', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({delta: deltaSec})
            }).then(response => response.json())
              .then(data => console.log('Seek:', data))
              .catch(error => console.error('Seek error:', error));
        }

        // 프로그레스 바 클릭으로 탐색
        function seekToPosition(event) {
            const progressBar = document.getElementById('progressBar');
            const rect = progressBar.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const percentage = clickX / rect.width;
            const targetTime = percentage * totalDuration;

            fetch('/seek', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({time: targetTime})
            }).then(response => response.json())
              .then(data => console.log('Seek to time:', data))
              .catch(error => console.error('Seek error:', error));
        }

        // 일시정지/재생 토글
        function togglePause() {
            fetch('/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            }).then(response => response.json())
              .then(data => {
                  isPaused = data.paused;
                  updateStats();
              })
              .catch(error => console.error('Pause error:', error));
        }

        // 키보드 단축키
        document.addEventListener('keydown', function(e) {
            if (!isVideoFile) return;

            switch(e.key) {
                case ' ':  // 스페이스바 = 일시정지/재생
                    e.preventDefault();
                    togglePause();
                    break;
                case 'ArrowLeft':  // 왼쪽 = -5초
                    e.preventDefault();
                    seek(-5);
                    break;
                case 'ArrowRight':  // 오른쪽 = +5초
                    e.preventDefault();
                    seek(5);
                    break;
                case 'ArrowUp':  // 위 = +10초
                    e.preventDefault();
                    seek(10);
                    break;
                case 'ArrowDown':  // 아래 = -10초
                    e.preventDefault();
                    seek(-10);
                    break;
            }
        });

        // 설정 로드
        function loadConfig() {
            fetch('/config')
                .then(response => response.json())
                .then(data => {
                    // 슬라이더 값 설정
                    if (data.detection_confidence !== undefined) {
                        document.getElementById('confidenceSlider').value = data.detection_confidence;
                        document.getElementById('confidenceValue').textContent = parseFloat(data.detection_confidence).toFixed(2);
                    }
                    document.getElementById('debugMode').checked = data.debug_mode || false;
                    updateIndicator('debugIndicator', data.debug_mode);
                    document.getElementById('enableDrawing').checked = data.enable_drawing;
                    updateIndicator('drawingIndicator', data.enable_drawing);
                    document.getElementById('frameSkipSlider').value = data.frame_skip || 1;
                    document.getElementById('frameSkipValue').textContent = data.frame_skip || 1;
                    document.getElementById('jpegQualitySlider').value = data.jpeg_quality || 70;
                    document.getElementById('jpegQualityValue').textContent = data.jpeg_quality || 70;
                    document.getElementById('resizeWidthSlider').value = data.resize_width || 640;
                    document.getElementById('resizeWidthValue').textContent = data.resize_width || 640;
                    
                    // 비디오 저장 상태
                    if (data.save_video) {
                        document.getElementById('startRecordBtn').disabled = true;
                        document.getElementById('stopRecordBtn').disabled = false;
                        document.getElementById('recordStatus').textContent = `저장 중: ${data.output_path || 'N/A'}`;
                        document.getElementById('recordStatus').style.color = '#4CAF50';
                    } else {
                        document.getElementById('startRecordBtn').disabled = false;
                        document.getElementById('stopRecordBtn').disabled = true;
                        document.getElementById('recordStatus').textContent = '';
                    }
                })
                .catch(error => console.error('설정 로드 오류:', error));
        }

        // 상태 표시기 업데이트
        function updateIndicator(id, active) {
            const indicator = document.getElementById(id);
            if (active) {
                indicator.classList.remove('inactive');
                indicator.classList.add('active');
            } else {
                indicator.classList.remove('active');
                indicator.classList.add('inactive');
            }
        }

        // 감지 신뢰도 업데이트
        function updateConfidence(value) {
            document.getElementById('confidenceValue').textContent = parseFloat(value).toFixed(2);
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({detection_confidence: parseFloat(value)})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('감지 신뢰도 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // 디버그 모드 업데이트
        function updateDebugMode(enabled) {
            updateIndicator('debugIndicator', enabled);
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({debug_mode: enabled})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('디버그 모드 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // 포즈 그리기 업데이트
        function updateDrawing(enabled) {
            updateIndicator('drawingIndicator', enabled);
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enable_drawing: enabled})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('포즈 그리기 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // 프레임 스킵 업데이트
        function updateFrameSkip(value) {
            document.getElementById('frameSkipValue').textContent = value;
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frame_skip: parseInt(value)})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('프레임 스킵 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // JPEG 품질 업데이트
        function updateJpegQuality(value) {
            document.getElementById('jpegQualityValue').textContent = value;
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({jpeg_quality: parseInt(value)})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('JPEG 품질 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // 리사이즈 너비 업데이트
        function updateResizeWidth(value) {
            const displayValue = value == 0 ? '원본' : value + 'px';
            document.getElementById('resizeWidthValue').textContent = displayValue;
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({resize_width: parseInt(value)})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('리사이즈 너비 업데이트:', data.changes);
                  }
              })
              .catch(error => console.error('설정 업데이트 오류:', error));
        }

        // 비디오 저장 시작
        function startVideoOutput() {
            const outputPath = document.getElementById('outputPath').value || 'output_video.mp4';
            fetch('/video_output', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'start', output_path: outputPath})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      document.getElementById('startRecordBtn').disabled = true;
                      document.getElementById('stopRecordBtn').disabled = false;
                      document.getElementById('recordStatus').textContent = `저장 중: ${data.output_path}`;
                      document.getElementById('recordStatus').style.color = '#4CAF50';
                      alert('비디오 저장이 시작되었습니다: ' + data.output_path);
                  } else {
                      alert('오류: ' + (data.error || '알 수 없는 오류'));
                  }
              })
              .catch(error => {
                  console.error('비디오 저장 시작 오류:', error);
                  alert('비디오 저장 시작 실패');
              });
        }

        // 비디오 저장 중지
        function stopVideoOutput() {
            fetch('/video_output', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'stop'})
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      document.getElementById('startRecordBtn').disabled = false;
                      document.getElementById('stopRecordBtn').disabled = true;
                      document.getElementById('recordStatus').textContent = `저장 완료: ${data.output_path}`;
                      document.getElementById('recordStatus').style.color = '#4CAF50';
                      alert('비디오 저장이 완료되었습니다: ' + data.output_path);
                      setTimeout(() => {
                          document.getElementById('recordStatus').textContent = '';
                      }, 5000);
                  } else {
                      alert('오류: ' + (data.error || '알 수 없는 오류'));
                  }
              })
              .catch(error => {
                  console.error('비디오 저장 중지 오류:', error);
                  alert('비디오 저장 중지 실패');
              });
        }

        // 통계 리셋
        function resetStats() {
            if (!confirm('통계를 리셋하시겠습니까?')) {
                return;
            }
            fetch('/reset_stats', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            }).then(response => response.json())
              .then(data => {
                  if (data.success) {
                      console.log('통계 리셋 완료');
                      updateStats(); // 즉시 통계 업데이트
                  }
              })
              .catch(error => console.error('통계 리셋 오류:', error));
        }

        // 통계 업데이트 함수 수정 (설정도 함께 업데이트)
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.processed_fps.toFixed(1);
                    document.getElementById('falls').textContent = data.fall_count;
                    document.getElementById('fights').textContent = data.fight_count;
                    document.getElementById('frames').textContent = data.frame_count;

                    // 경고 표시
                    const alert = document.getElementById('alert');
                    if (data.fall_count > 0 || data.fight_count > 0) {
                        alert.classList.add('show');
                    } else {
                        alert.classList.remove('show');
                    }

                    // 비디오 컨트롤 표시/업데이트
                    isVideoFile = data.is_video_file;
                    if (isVideoFile) {
                        document.getElementById('videoControls').classList.add('show');
                        totalDuration = data.duration;

                        // 프로그레스 바 업데이트
                        const progress = (data.current_time / data.duration) * 100;
                        document.getElementById('progressFill').style.width = progress + '%';
                        document.getElementById('timeDisplay').textContent =
                            formatTime(data.current_time) + ' / ' + formatTime(data.duration);

                        // 일시정지 버튼 상태
                        isPaused = data.paused;
                        const pauseBtn = document.getElementById('pauseBtn');
                        if (isPaused) {
                            pauseBtn.textContent = '▶ 재생';
                            pauseBtn.classList.add('active');
                        } else {
                            pauseBtn.textContent = '⏸ 일시정지';
                            pauseBtn.classList.remove('active');
                        }
                    }
                })
                .catch(error => console.error('통계 업데이트 오류:', error));
        }

        // 500ms마다 통계 업데이트
        setInterval(updateStats, 500);
        
        // 초기 로드
        updateStats();
        loadConfig();
        
        // 설정도 주기적으로 업데이트 (비디오 저장 상태 확인용)
        setInterval(loadConfig, 2000);
    </script>
</body>
</html>
"""

