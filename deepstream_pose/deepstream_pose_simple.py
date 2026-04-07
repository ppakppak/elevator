#!/usr/bin/env python3
"""
DeepStream Human Pose Estimation - 간소화 버전
기본 사람 감지 + Python 기반 포즈 분석
"""

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds

# 상수
MUXER_OUTPUT_WIDTH = 1280
MUXER_OUTPUT_HEIGHT = 720
MUXER_BATCH_TIMEOUT_USEC = 40000

# DeepStream 모델 설정 파일 경로 (절대 경로)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PGIE_CONFIG_FILE = "/home/ppak/projects/elevator/deepstream_pose/config/config_infer_primary.txt"
ENGINE_FILE = "/home/ppak/projects/elevator/deepstream_pose/models/Primary_Detector/resnet10.caffemodel_b1_gpu0_int8.engine"


class PersonTracker:
    """사람 추적 및 이벤트 감지"""

    def __init__(self):
        self.previous_boxes = []
        self.fall_count = 0
        self.fight_count = 0
        self.FIGHT_DISTANCE_THRESHOLD = 150
        self.FALL_ASPECT_RATIO_THRESHOLD = 1.5

    def detect_fall_by_bbox(self, bbox):
        left, top, width, height = bbox
        aspect_ratio = width / max(height, 1)
        if aspect_ratio > self.FALL_ASPECT_RATIO_THRESHOLD:
            return True, min(aspect_ratio / 3.0, 1.0)
        return False, 0.0

    def detect_fighting(self, boxes):
        if len(boxes) < 2:
            return False, 0.0
        max_score = 0.0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i + 1:]:
                c1 = (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
                c2 = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)
                dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                if dist < self.FIGHT_DISTANCE_THRESHOLD:
                    score = 1.0 - (dist / self.FIGHT_DISTANCE_THRESHOLD)
                    max_score = max(max_score, score)
        return max_score > 0.5, max_score


class DeepStreamApp:
    """DeepStream 사람 감지 애플리케이션"""

    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.tracker = PersonTracker()
        self.frame_count = 0
        self.start_time = time.time()
        self.overlay_json = Path(args.overlay_json).expanduser() if args.overlay_json else None
        self.overlay_interval_sec = max(float(args.overlay_interval_ms), 50.0) / 1000.0
        self._last_overlay_write_ts = 0.0
        if self.overlay_json:
            self.overlay_json.parent.mkdir(parents=True, exist_ok=True)
        Gst.init(None)

    def _write_overlay(self, payload, force=False):
        if not self.overlay_json:
            return
        now = time.time()
        if not force and (now - self._last_overlay_write_ts) < self.overlay_interval_sec:
            return
        payload = dict(payload)
        payload.setdefault('updated_at', now)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=str(self.overlay_json.parent), suffix='.tmp') as tmp:
                json.dump(payload, tmp, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.overlay_json)
            self._last_overlay_write_ts = now
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _publish_status(self, active, **extra):
        payload = {
            'active': active,
            'source': self.args.source,
            'frame_count': self.frame_count,
            'person_count': 0,
            'fall_count': self.tracker.fall_count,
            'fight_count': self.tracker.fight_count,
            'fight_active': False,
            'boxes': [],
        }
        payload.update(extra)
        self._write_overlay(payload, force=True)

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_width = MUXER_OUTPUT_WIDTH
            frame_height = MUXER_OUTPUT_HEIGHT
            overlay_boxes = []
            person_boxes = []
            fallen_now = 0
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.class_id in [0, 2]:
                    bbox = (
                        float(obj_meta.rect_params.left),
                        float(obj_meta.rect_params.top),
                        float(obj_meta.rect_params.width),
                        float(obj_meta.rect_params.height),
                    )
                    person_boxes.append(bbox)

                    is_fallen, fall_conf = self.tracker.detect_fall_by_bbox(bbox)
                    if is_fallen:
                        fallen_now += 1
                        self.tracker.fall_count += 1
                        obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 4
                        print(f"[쓰러짐 감지] 프레임 {self.frame_count}, 신뢰도: {fall_conf:.2f}")
                    else:
                        obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 3

                    overlay_boxes.append({
                        'left': bbox[0],
                        'top': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3],
                        'fallen': is_fallen,
                        'score': round(float(fall_conf if is_fallen else getattr(obj_meta, 'confidence', 0.0) or 0.0), 3),
                    })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            is_fighting, fight_conf = self.tracker.detect_fighting(person_boxes)
            if is_fighting:
                self.tracker.fight_count += 1
                print(f"[싸움 감지] 프레임 {self.frame_count}, 신뢰도: {fight_conf:.2f}")

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1

            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            py_nvosd_text_params = display_meta.text_params[0]
            py_nvosd_text_params.display_text = (
                f"FPS: {fps:.1f} | Persons: {len(person_boxes)} | "
                f"Falls: {self.tracker.fall_count} | Fights: {self.tracker.fight_count}"
            )
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12
            py_nvosd_text_params.font_params.font_name = 'Serif'
            py_nvosd_text_params.font_params.font_size = 12
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            self._write_overlay({
                'active': True,
                'source': self.args.source,
                'frame_count': self.frame_count,
                'source_width': frame_width,
                'source_height': frame_height,
                'person_count': len(person_boxes),
                'fallen_now': fallen_now,
                'fall_count': self.tracker.fall_count,
                'fight_count': self.tracker.fight_count,
                'fight_active': is_fighting,
                'fight_confidence': round(float(fight_conf), 3),
                'fps': round(fps, 2),
                'boxes': overlay_boxes,
            })

            self.frame_count += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n스트림 종료")
            self._publish_status(False, note='eos')
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"경고: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"오류: {err}: {debug}")
            self._publish_status(False, note=f'error: {err}')
            loop.quit()
        return True

    def create_pipeline(self):
        print('파이프라인 생성 중...')

        source = self.args.source
        src_lower = source.lower()
        is_live_uri = src_lower.startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://"))

        if source.isdigit():
            sink_str = 'fakesink sync=1' if self.args.no_display else 'xvimagesink sync=0'
            pipeline_str = f"""
                v4l2src device=/dev/video{source} !
                videoconvert !
                nvvideoconvert !
                video/x-raw(memory:NVMM), format=NV12 !
                m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720
                    batched-push-timeout={MUXER_BATCH_TIMEOUT_USEC} live-source=1 !
                nvinfer config-file-path={PGIE_CONFIG_FILE} !
                nvvideoconvert !
                nvdsosd name=osd !
                nvvideoconvert !
                video/x-raw, format=BGRx !
                videoconvert !
                {sink_str}
            """
        else:
            sink_str = 'fakesink sync=1' if self.args.no_display else 'xvimagesink sync=0'
            uri = source
            if '://' not in source and os.path.exists(source):
                uri = 'file://' + os.path.abspath(source)
            live_flag = 1 if is_live_uri else 0
            pipeline_str = f"""
                uridecodebin uri={uri} !
                nvvideoconvert !
                video/x-raw(memory:NVMM), format=NV12 !
                m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720
                    batched-push-timeout={MUXER_BATCH_TIMEOUT_USEC} live-source={live_flag} !
                nvinfer config-file-path={PGIE_CONFIG_FILE} !
                nvvideoconvert !
                nvdsosd name=osd !
                nvvideoconvert !
                video/x-raw, format=BGRx !
                videoconvert !
                {sink_str}
            """

        print(f"파이프라인: {pipeline_str.strip()}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        if not self.pipeline:
            raise RuntimeError('파이프라인 생성 실패')

        osd = self.pipeline.get_by_name('osd')
        if osd:
            osdsinkpad = osd.get_static_pad('sink')
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print('파이프라인 생성 완료')

    def run(self):
        print('=' * 60)
        print('DeepStream Person Detection (Simple Version)')
        print('=' * 60)
        print(f'소스: {self.args.source}')
        print(f"화면 표시: {'비활성화' if self.args.no_display else '활성화'}")
        if self.overlay_json:
            print(f'오버레이 JSON: {self.overlay_json}')
        print('=' * 60)

        self._publish_status(False, note='starting')
        self.create_pipeline()

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.bus_call, self.loop)

        print('파이프라인 시작...')
        self.pipeline.set_state(Gst.State.PLAYING)
        self._publish_status(True, note='running')

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('\n사용자에 의해 중단됨')
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self._publish_status(False, note='stopped')
            elapsed = time.time() - self.start_time
            print('=' * 60)
            print('최종 통계:')
            print(f'  총 프레임: {self.frame_count}')
            print(f'  처리 시간: {elapsed:.2f}초')
            if elapsed > 0:
                print(f'  평균 FPS: {self.frame_count / elapsed:.2f}')
            print(f'  쓰러짐 감지: {self.tracker.fall_count}')
            print(f'  싸움 감지: {self.tracker.fight_count}')
            print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='DeepStream Person Detection')
    parser.add_argument('--source', type=str, default='0', help='비디오 소스 (카메라 번호 또는 URI)')
    parser.add_argument('--no-display', action='store_true', help='화면 표시 비활성화')
    parser.add_argument('--overlay-json', help='현재 추론 메타데이터를 기록할 JSON 파일 경로')
    parser.add_argument('--overlay-interval-ms', type=float, default=120.0, help='오버레이 JSON 기록 주기(ms)')

    args = parser.parse_args()
    app = DeepStreamApp(args)
    app.run()


if __name__ == '__main__':
    main()
