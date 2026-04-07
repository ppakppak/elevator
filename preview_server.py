#!/usr/bin/env python3
import argparse
import json
import time
import threading
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify


class PreviewServer:
    def __init__(self, source, host='0.0.0.0', port=5000, width=640, jpeg_quality=70, overlay_json=None):
        self.source = source
        self.host = host
        self.port = port
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.overlay_json = Path(overlay_json).expanduser() if overlay_json else None
        self.app = Flask(__name__)
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        self._overlay_cache = None
        self._overlay_mtime_ns = None

        src = str(source).lower()
        self.is_live = src.isdigit() or src.startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://"))

        self._setup_routes()

    def _open_capture(self):
        src = self.source
        if str(src).isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

    def _load_overlay(self):
        if not self.overlay_json:
            return None
        try:
            if not self.overlay_json.exists():
                return self._overlay_cache
            stat = self.overlay_json.stat()
            if self._overlay_mtime_ns != stat.st_mtime_ns:
                self._overlay_cache = json.loads(self.overlay_json.read_text())
                self._overlay_mtime_ns = stat.st_mtime_ns
            return self._overlay_cache
        except Exception:
            return self._overlay_cache

    def _draw_overlay(self, frame):
        overlay = self._load_overlay()
        if not overlay:
            return frame

        result = frame.copy()
        now = time.time()
        updated_at = float(overlay.get('updated_at', 0) or 0)
        is_fresh = updated_at > 0 and (now - updated_at) < 2.0
        source_width = max(int(overlay.get('source_width') or frame.shape[1]), 1)
        source_height = max(int(overlay.get('source_height') or frame.shape[0]), 1)
        scale_x = frame.shape[1] / source_width
        scale_y = frame.shape[0] / source_height

        if is_fresh:
            for box in overlay.get('boxes', []):
                left = int(max(0, min(frame.shape[1] - 1, round(float(box.get('left', 0)) * scale_x))))
                top = int(max(0, min(frame.shape[0] - 1, round(float(box.get('top', 0)) * scale_y))))
                width = int(max(1, round(float(box.get('width', 0)) * scale_x)))
                height = int(max(1, round(float(box.get('height', 0)) * scale_y)))
                right = min(frame.shape[1] - 1, left + width)
                bottom = min(frame.shape[0] - 1, top + height)
                fallen = bool(box.get('fallen'))
                color = (0, 0, 255) if fallen else (0, 200, 0)
                label = 'FALL' if fallen else 'PERSON'
                score = box.get('score')
                if isinstance(score, (int, float)):
                    label = f"{label} {score:.2f}"
                cv2.rectangle(result, (left, top), (right, bottom), color, 2)
                cv2.putText(result, label, (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        fight_active = bool(overlay.get('fight_active')) and is_fresh
        person_count = int(overlay.get('person_count', 0) or 0)
        fallen_now = int(overlay.get('fallen_now', 0) or 0)
        status_color = (0, 200, 0) if is_fresh else (0, 165, 255)
        if fight_active:
            status_color = (0, 0, 255)

        cv2.rectangle(result, (0, 0), (result.shape[1], 32), (18, 18, 18), -1)
        ai_state = 'AI ON' if is_fresh else 'AI STALE'
        banner = f"{ai_state} | Persons:{person_count} | Fallen:{fallen_now}"
        if fight_active:
            banner += ' | FIGHT ALERT'
        cv2.putText(result, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        return result

    def _reader(self):
        self._open_capture()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                self._open_capture()
                continue

            ok, frame = self.cap.read()
            if not ok:
                if not self.is_live and self.cap is not None and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                time.sleep(0.02)
                continue

            if self.width and frame.shape[1] > self.width:
                h = int(frame.shape[0] * self.width / frame.shape[1])
                frame = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_LINEAR)

            frame = self._draw_overlay(frame)

            with self.lock:
                self.frame = frame
                self.frame_count += 1

            if self.is_live:
                time.sleep(0.001)
            else:
                time.sleep(1 / 15)

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            overlay_text = 'overlay on' if self.overlay_json else 'overlay off'
            return (
                '<html><head><title>Elevator Preview</title></head>'
                '<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">'
                f'<div style="padding:10px">Elevator Preview ({overlay_text})</div>'
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
            overlay = self._load_overlay() or {}
            updated_at = float(overlay.get('updated_at', 0) or 0)
            return jsonify({
                'frames': self.frame_count,
                'fps': round(fps, 2),
                'source': self.source,
                'overlay_enabled': bool(self.overlay_json),
                'overlay_fresh': updated_at > 0 and (time.time() - updated_at) < 2.0,
                'person_count': int(overlay.get('person_count', 0) or 0),
                'fight_active': bool(overlay.get('fight_active', False)),
            })

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
    p.add_argument('--overlay-json')
    args = p.parse_args()

    PreviewServer(
        source=args.source,
        host=args.host,
        port=args.port,
        width=args.width,
        jpeg_quality=args.jpeg_quality,
        overlay_json=args.overlay_json,
    ).run()


if __name__ == '__main__':
    main()
