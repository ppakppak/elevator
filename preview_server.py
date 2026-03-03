#!/usr/bin/env python3
import argparse
import time
import threading
import cv2
from flask import Flask, Response, jsonify


class PreviewServer:
    def __init__(self, source, host='0.0.0.0', port=5000, width=640, jpeg_quality=70):
        self.source = source
        self.host = host
        self.port = port
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.app = Flask(__name__)
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()

        src = str(source).lower()
        self.is_live = src.isdigit() or src.startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://"))

        self._setup_routes()

    def _open_capture(self):
        src = self.source
        if str(src).isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        # low-latency hints (backend-dependent)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

    def _reader(self):
        self._open_capture()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                self._open_capture()
                continue

            ok, frame = self.cap.read()
            if not ok:
                # 파일 소스는 EOF 도달 시 처음으로 되감기
                if not self.is_live and self.cap is not None and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue

                time.sleep(0.02)
                continue

            if self.width and frame.shape[1] > self.width:
                h = int(frame.shape[0] * self.width / frame.shape[1])
                frame = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_LINEAR)

            with self.lock:
                self.frame = frame
                self.frame_count += 1

            # 파일 재생은 과도한 CPU 사용 방지를 위해 속도 제한
            if self.is_live:
                time.sleep(0.001)
            else:
                time.sleep(1/15)

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return (
                '<html><head><title>Elevator Preview</title></head>'
                '<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">'
                '<div style="padding:10px">Elevator Preview (DeepStream inference running separately)</div>'
                '<img src="/video_feed" style="width:100%;height:auto;display:block"/>'
                '</body></html>'
            )

        @self.app.route('/video_feed')
        def video_feed():
            def gen():
                last = None
                while self.running:
                    cur = None
                    with self.lock:
                        if self.frame is not None:
                            ok, buf = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                            if ok:
                                cur = buf.tobytes()
                                last = cur
                    if cur is None and last is not None:
                        cur = last
                    if cur is not None:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cur + b'\r\n')
                    time.sleep(0.03)

            return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/stats')
        def stats():
            elapsed = max(time.time() - self.start_time, 1e-6)
            fps = self.frame_count / elapsed
            return jsonify({'frames': self.frame_count, 'fps': round(fps, 2), 'source': self.source})

    def run(self):
        self.running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True, use_reloader=False)


def main():
    p = argparse.ArgumentParser(description='Low-latency preview server')
    p.add_argument('--source', required=True)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--jpeg-quality', type=int, default=70)
    args = p.parse_args()

    PreviewServer(
        source=args.source,
        host=args.host,
        port=args.port,
        width=args.width,
        jpeg_quality=args.jpeg_quality,
    ).run()


if __name__ == '__main__':
    main()
