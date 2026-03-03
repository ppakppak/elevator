#!/usr/bin/env python3
"""
DeepStream Human Pose Estimation - 간소화 버전
기본 사람 감지 + Python 기반 포즈 분석
"""

import sys
import os
import time
import argparse
import math

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
        """바운딩 박스 기반 쓰러짐 감지"""
        left, top, width, height = bbox
        aspect_ratio = width / max(height, 1)
        if aspect_ratio > self.FALL_ASPECT_RATIO_THRESHOLD:
            return True, min(aspect_ratio / 3.0, 1.0)
        return False, 0.0

    def detect_fighting(self, boxes):
        """근접도 기반 싸움 감지"""
        if len(boxes) < 2:
            return False, 0.0
        max_score = 0.0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                c1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
                c2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
                dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
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
        Gst.init(None)

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """버퍼 프로브 - 메타데이터 처리"""
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

            person_boxes = []
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.class_id in [0, 2]:
                    bbox = (
                        obj_meta.rect_params.left,
                        obj_meta.rect_params.top,
                        obj_meta.rect_params.width,
                        obj_meta.rect_params.height
                    )
                    person_boxes.append(bbox)

                    is_fallen, fall_conf = self.tracker.detect_fall_by_bbox(bbox)
                    if is_fallen:
                        self.tracker.fall_count += 1
                        obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 4
                        print(f"[쓰러짐 감지] 프레임 {self.frame_count}, 신뢰도: {fall_conf:.2f}")

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            is_fighting, fight_conf = self.tracker.detect_fighting(person_boxes)
            if is_fighting:
                self.tracker.fight_count += 1
                print(f"[싸움 감지] 프레임 {self.frame_count}, 신뢰도: {fight_conf:.2f}")

            # 화면에 통계 표시
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
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 12
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            self.frame_count += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def bus_call(self, bus, message, loop):
        """GStreamer 버스 메시지 핸들러"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n스트림 종료")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"경고: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"오류: {err}: {debug}")
            loop.quit()
        return True

    def create_pipeline(self):
        """파이프라인 생성"""
        print("파이프라인 생성 중...")

        # GStreamer 파이프라인 문자열 생성
        source = self.args.source
        src_lower = source.lower()
        is_live_uri = src_lower.startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://"))

        if source.isdigit():
            # USB 카메라
            sink_str = "fakesink sync=1" if self.args.no_display else "xvimagesink sync=0"
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
            # 파일/URI
            if self.args.no_display:
                # 파일 소스 과속 방지를 위해 sync=1로 실시간에 가깝게 동작
                sink_str = "fakesink sync=1"
            else:
                sink_str = "xvimagesink sync=0"

            # 로컬 파일 경로면 file:// URI로 변환
            uri = source
            if "://" not in source and os.path.exists(source):
                uri = "file://" + os.path.abspath(source)

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
            raise RuntimeError("파이프라인 생성 실패")

        # OSD 프로브 추가
        osd = self.pipeline.get_by_name("osd")
        if osd:
            osdsinkpad = osd.get_static_pad("sink")
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print("파이프라인 생성 완료")

    def run(self):
        """실행"""
        print("=" * 60)
        print("DeepStream Person Detection (Simple Version)")
        print("=" * 60)
        print(f"소스: {self.args.source}")
        print(f"화면 표시: {'비활성화' if self.args.no_display else '활성화'}")
        print("=" * 60)

        self.create_pipeline()

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        print("파이프라인 시작...")
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        finally:
            self.pipeline.set_state(Gst.State.NULL)

            elapsed = time.time() - self.start_time
            print("=" * 60)
            print("최종 통계:")
            print(f"  총 프레임: {self.frame_count}")
            print(f"  처리 시간: {elapsed:.2f}초")
            if elapsed > 0:
                print(f"  평균 FPS: {self.frame_count / elapsed:.2f}")
            print(f"  쓰러짐 감지: {self.tracker.fall_count}")
            print(f"  싸움 감지: {self.tracker.fight_count}")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DeepStream Person Detection")
    parser.add_argument('--source', type=str, default='0',
                       help='비디오 소스 (카메라 번호 또는 URI)')
    parser.add_argument('--no-display', action='store_true',
                       help='화면 표시 비활성화')

    args = parser.parse_args()
    app = DeepStreamApp(args)
    app.run()


if __name__ == "__main__":
    main()
