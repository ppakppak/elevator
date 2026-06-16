#!/usr/bin/env python3
"""
Systemd-friendly YOLO segmentation runner for staged dashboard rollout.

This keeps the overlay JSON compatible with preview_server.py so operators can
switch a single channel to best_m without touching the preview path.
"""

import argparse
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import cv2

from detector_segmentation import create_detector

OVERLAY_INTERVAL_SEC = 0.05  # faster preview sync for file sources
FILE_PLAYBACK_FPS = 15.0  # mirror preview_server.py for file playback sync
LIVE_SCHEMES = ("rtsp://", "rtmp://", "udp://", "http://", "https://")


def _is_live_source(source: str) -> bool:
    src = str(source).strip().lower()
    return src.isdigit() or src.startswith(LIVE_SCHEMES)


class SegmentationServiceRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.overlay_json = Path(args.overlay_json).expanduser() if args.overlay_json else None
        if self.overlay_json:
            self.overlay_json.parent.mkdir(parents=True, exist_ok=True)

        self.detector = create_detector(
            model_size=args.seg_model,
            confidence=args.conf,
            device=args.device,
            imgsz=args.imgsz,
        )

        self.running = True
        self.cap = None
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.fall_count = 0
        self.fight_count = 0
        self._last_overlay_write_ts = 0.0
        self.is_live = _is_live_source(args.source)
        self.file_frame_interval_sec = 0.0 if self.is_live else (1.0 / FILE_PLAYBACK_FPS)
        self._next_frame_due_ts = None
        self.resize_width = args.resize_width
        self.preview_frame_path = None
        if self.overlay_json:
            self.preview_frame_path = self.overlay_json.parent.parent / "frames" / f"{self.overlay_json.stem}.jpg"
            self.preview_frame_path.parent.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum, _frame):
        print(f"종료 신호 수신: {signum}", flush=True)
        self.running = False

    def _open_capture(self):
        source = int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            raise RuntimeError(f"비디오 소스를 열 수 없습니다: {self.args.source}")
        return cap

    def _write_overlay(self, payload, force=False):
        if not self.overlay_json:
            return
        now = time.time()
        if not force and (now - self._last_overlay_write_ts) < OVERLAY_INTERVAL_SEC:
            return
        payload = dict(payload)
        payload.setdefault("updated_at", now)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=str(self.overlay_json.parent), suffix=".tmp"
            ) as tmp:
                json.dump(payload, tmp, ensure_ascii=False)
                tmp.flush()
                # No fsync here: this file is a live preview/status handoff, not durable state.
                # fsync per frame throttles Jetson file-source preview and AI/overlay sync.
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.overlay_json)
            self._last_overlay_write_ts = now
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _sync_file_playback(self):
        if self.file_frame_interval_sec <= 0:
            return
        now = time.monotonic()
        if self._next_frame_due_ts is None:
            self._next_frame_due_ts = now + self.file_frame_interval_sec
            return
        sleep_for = self._next_frame_due_ts - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.monotonic()
        self._next_frame_due_ts = max(self._next_frame_due_ts + self.file_frame_interval_sec, now)

    def _write_preview_frame(self, frame):
        if self.preview_frame_path is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", delete=False, dir=str(self.preview_frame_path.parent), suffix=".jpg.tmp"
            ) as tmp:
                tmp.write(buf.tobytes())
                tmp.flush()
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.preview_frame_path)
            return str(self.preview_frame_path)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _publish_status(self, active: bool, **extra):
        payload = {
            "active": active,
            "source": self.args.source,
            "frame_count": self.processed_count,
            "person_count": 0,
            "fallen_now": 0,
            "fall_count": self.fall_count,
            "fight_count": self.fight_count,
            "fight_active": False,
            "boxes": [],
        }
        payload.update(extra)
        self._write_overlay(payload, force=True)

    def _process_frame(self, frame, source_frame_index=None):
        frame_height, frame_width = frame.shape[:2]
        detections = self.detector.process_frame(frame, frame_number=self.frame_count)

        person_count = 0
        fallen_now = 0
        fight_now = 0
        max_fight_conf = 0.0
        boxes = []

        for det in detections:
            x, y, w, h = det.bbox
            person_count += 1

            if det.is_fallen:
                fallen_now += 1
                self.fall_count += 1
                print(f"[쓰러짐 감지] 프레임 {self.frame_count}, 신뢰도: {det.confidence:.2f}", flush=True)

            if det.is_fighting:
                fight_now += 1
                self.fight_count += 1
                max_fight_conf = max(max_fight_conf, float(det.confidence))
                print(f"[싸움 감지] 프레임 {self.frame_count}, 신뢰도: {det.confidence:.2f}", flush=True)

            boxes.append(
                {
                    "left": float(x),
                    "top": float(y),
                    "width": float(w),
                    "height": float(h),
                    "fallen": bool(det.is_fallen),
                    "fighting": bool(det.is_fighting),
                    "score": round(float(det.confidence), 3),
                    "class_id": int(det.class_id),
                    "class_name": det.class_name,
                }
            )

        self.processed_count += 1
        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = self.processed_count / elapsed
        preview_frame_jpeg = self._write_preview_frame(frame)
        payload = {
            "active": True,
            "source": self.args.source,
            "frame_count": self.processed_count,
            "source_frame_index": int(source_frame_index) if source_frame_index is not None else self.frame_count,
            "source_width": frame_width,
            "source_height": frame_height,
            "person_count": person_count,
            "fallen_now": fallen_now,
            "fall_count": self.fall_count,
            "fight_count": self.fight_count,
            "fight_active": fight_now > 0,
            "fight_confidence": round(max_fight_conf, 3),
            "fps": round(fps, 2),
            "boxes": boxes,
        }
        if preview_frame_jpeg:
            payload["preview_frame_jpeg"] = preview_frame_jpeg
        self._write_overlay(payload)

    def run(self):
        self.cap = self._open_capture()
        self._publish_status(True, note="starting")
        print(
            f"세그멘테이션 서비스 시작: source={self.args.source} model=best_{self.args.seg_model}",
            flush=True,
        )

        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    if not self.running:
                        break
                    if not self.is_live:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue

                    print(f"경고: 프레임 읽기 실패, 재연결 시도: {self.args.source}", flush=True)
                    self._publish_status(False, note="read_failed")
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = None
                    while self.running and self.cap is None:
                        try:
                            time.sleep(1.0)
                            self.cap = self._open_capture()
                            self._publish_status(True, note="reconnected")
                        except Exception as exc:
                            print(f"경고: 소스 재연결 실패: {exc}", flush=True)
                    continue

                # 프레임 리사이즈 (전처리/후처리 가속)
                if self.resize_width > 0 and frame.shape[1] > self.resize_width:
                    scale = self.resize_width / frame.shape[1]
                    frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * scale)))

                self.frame_count += 1
                source_frame_index = None
                if not self.is_live and self.cap is not None:
                    try:
                        source_frame_index = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                    except Exception:
                        source_frame_index = None
                if self.args.frame_skip > 1 and (self.frame_count % self.args.frame_skip) != 0:
                    continue

                self._process_frame(frame, source_frame_index=source_frame_index)
        finally:
            self._publish_status(False, note="stopped")
            if self.cap is not None:
                self.cap.release()
            self.detector.release()
            print("세그멘테이션 서비스 종료", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Dashboard segmentation inference runner")
    parser.add_argument("--source", required=True, help="webcam index, RTSP/HTTP URL, or file path")
    parser.add_argument("--overlay-json", required=True, help="overlay JSON path consumed by preview_server.py")
    parser.add_argument("--frame-skip", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--conf", type=float, default=0.3, help="minimum detection confidence")
    parser.add_argument("--device", default="auto", help="inference device (auto, cpu, cuda, 0, ...)")
    parser.add_argument("--imgsz", type=int, default=384, help="YOLO segmentation inference size")
    parser.add_argument("--seg-model", default="m", choices=["m", "l"], help="best_<size> model variant")
    parser.add_argument("--resize-width", type=int, default=640, help="resize frame width before inference (0=no resize)")
    return parser


def main():
    args = build_parser().parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")
    if not 0.0 <= float(args.conf) <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")

    runner = SegmentationServiceRunner(args)
    try:
        runner.run()
    except RuntimeError as exc:
        runner._publish_status(False, note=str(exc))
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
