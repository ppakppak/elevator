#!/usr/bin/env python3
"""
DeepStream Human Pose Estimation Application
GStreamer 기반 실시간 포즈 추정 및 이벤트 감지

특징:
- trt_pose 모델을 이용한 고성능 포즈 추정
- TensorRT 가속을 통한 실시간 처리
- 쓰러짐/싸움 감지 기능
- 다중 소스 지원 (카메라, 파일, RTSP)
"""

import sys
import os
import time
import argparse
import math
from typing import List, Tuple, Optional, Dict, Any

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

import numpy as np
import pyds  # DeepStream Python bindings

# CUDA/cuDNN 경고 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 상수
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1280
MUXER_OUTPUT_HEIGHT = 720
MUXER_BATCH_TIMEOUT_USEC = 40000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720

# 포즈 키포인트 인덱스 (COCO)
class KeypointIndex:
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

# 스켈레톤 연결 및 색상
SKELETON_CONNECTIONS = [
    (KeypointIndex.LEFT_ANKLE, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_ANKLE, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    (KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
    (KeypointIndex.NOSE, KeypointIndex.NECK),
    (KeypointIndex.NECK, KeypointIndex.LEFT_SHOULDER),
    (KeypointIndex.NECK, KeypointIndex.RIGHT_SHOULDER),
]


class PersonPose:
    """개별 포즈 정보"""
    def __init__(self):
        self.keypoints: List[Tuple[float, float, float]] = [(0, 0, 0)] * 18
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.score: float = 0.0
        self.is_fallen: bool = False
        self.fall_confidence: float = 0.0


class PoseAnalyzer:
    """포즈 분석 및 이벤트 감지"""

    def __init__(self):
        self.previous_poses: List[PersonPose] = []

        # 쓰러짐 감지 임계값
        self.FALL_ANGLE_THRESHOLD = 45
        self.FALL_HEIGHT_RATIO_THRESHOLD = 0.3

        # 싸움 감지 임계값
        self.FIGHT_DISTANCE_THRESHOLD = 150
        self.FIGHT_MOVEMENT_THRESHOLD = 30

    def get_keypoint(self, keypoints, idx: int) -> Optional[Tuple[float, float]]:
        """키포인트 가져오기"""
        if idx < len(keypoints) and keypoints[idx][2] > 0.5:
            return (keypoints[idx][0], keypoints[idx][1])
        return None

    def detect_fall(self, pose: PersonPose, frame_width: int, frame_height: int) -> Tuple[bool, float]:
        """쓰러짐 감지"""
        keypoints = pose.keypoints

        left_shoulder = self.get_keypoint(keypoints, KeypointIndex.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(keypoints, KeypointIndex.RIGHT_SHOULDER)
        left_hip = self.get_keypoint(keypoints, KeypointIndex.LEFT_HIP)
        right_hip = self.get_keypoint(keypoints, KeypointIndex.RIGHT_HIP)

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return False, 0.0

        # 정규화된 좌표로 변환
        ls = (left_shoulder[0] / frame_width, left_shoulder[1] / frame_height)
        rs = (right_shoulder[0] / frame_width, right_shoulder[1] / frame_height)
        lh = (left_hip[0] / frame_width, left_hip[1] / frame_height)
        rh = (right_hip[0] / frame_width, right_hip[1] / frame_height)

        # 어깨와 엉덩이의 중심점
        shoulder_center = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_center = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        # 높이 차이 기반 점수
        height_diff = abs(shoulder_center[1] - hip_center[1])
        height_score = 1.0 - min(height_diff / 0.1, 1.0) if height_diff < 0.1 else 0.0

        # 수평 위치 점수
        horizontal_score = 1.0 if abs(shoulder_center[1] - hip_center[1]) < 0.05 else 0.0

        # 종합 신뢰도
        confidence = height_score * 0.5 + horizontal_score * 0.5
        is_fallen = confidence > 0.5

        return is_fallen, confidence

    def detect_fighting(
        self,
        current_poses: List[PersonPose],
        frame_width: int,
        frame_height: int
    ) -> Tuple[bool, float]:
        """싸움 감지"""
        if len(current_poses) < 2:
            return False, 0.0

        fighting_score = 0.0

        # 거리 기반 점수
        for i, pose1 in enumerate(current_poses):
            for pose2 in current_poses[i+1:]:
                # 바운딩 박스 중심 간 거리
                c1 = (pose1.bbox[0] + pose1.bbox[2] / 2,
                      pose1.bbox[1] + pose1.bbox[3] / 2)
                c2 = (pose2.bbox[0] + pose2.bbox[2] / 2,
                      pose2.bbox[1] + pose2.bbox[3] / 2)

                dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

                if dist < self.FIGHT_DISTANCE_THRESHOLD:
                    distance_score = 1.0 - (dist / self.FIGHT_DISTANCE_THRESHOLD)
                    fighting_score = max(fighting_score, distance_score * 0.5)

        # 쓰러짐 연계 점수
        fallen_count = sum(1 for p in current_poses if p.is_fallen)
        if fallen_count > 0 and len(current_poses) >= 2:
            fighting_score += 0.3

        is_fighting = fighting_score > 0.5
        return is_fighting, fighting_score

    def analyze(
        self,
        poses: List[PersonPose],
        frame_width: int,
        frame_height: int
    ) -> Tuple[List[PersonPose], bool, float]:
        """전체 분석"""
        # 쓰러짐 감지
        for pose in poses:
            pose.is_fallen, pose.fall_confidence = self.detect_fall(
                pose, frame_width, frame_height
            )

        # 싸움 감지
        is_fighting, fight_confidence = self.detect_fighting(
            poses, frame_width, frame_height
        )

        self.previous_poses = poses
        return poses, is_fighting, fight_confidence


class DeepStreamPoseApp:
    """DeepStream Pose Estimation 애플리케이션"""

    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.analyzer = PoseAnalyzer()
        self.frame_count = 0
        self.start_time = time.time()
        self.fall_count = 0
        self.fight_count = 0

        # GStreamer 초기화
        Gst.init(None)

    def create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        """소스 빈 생성"""
        bin_name = f"source-bin-{index:02d}"

        # 카메라 또는 URI 소스 구분
        if uri.isdigit():
            # V4L2 카메라
            source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
            source.set_property("device", f"/dev/video{uri}")

            caps = Gst.ElementFactory.make("capsfilter", "source-caps")
            caps.set_property("caps", Gst.Caps.from_string(
                "video/x-raw, width=1280, height=720, framerate=30/1"
            ))

            vidconv = Gst.ElementFactory.make("videoconvert", "src-vidconv")
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "src-nvvidconv")

            nv_bin = Gst.Bin.new(bin_name)
            nv_bin.add(source)
            nv_bin.add(caps)
            nv_bin.add(vidconv)
            nv_bin.add(nvvidconv)

            source.link(caps)
            caps.link(vidconv)
            vidconv.link(nvvidconv)

            # Ghost pad 생성
            srcpad = nvvidconv.get_static_pad("src")
            ghost_pad = Gst.GhostPad.new("src", srcpad)
            nv_bin.add_pad(ghost_pad)

            return nv_bin

        else:
            # URI 소스 (파일, RTSP 등)
            nv_bin = Gst.Bin.new(bin_name)

            uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
            uri_decode_bin.set_property("uri", uri)

            # 동적 패드 연결
            uri_decode_bin.connect("pad-added", self.cb_newpad, nv_bin)
            uri_decode_bin.connect("child-added", self.cb_child_added, nv_bin)

            nv_bin.add(uri_decode_bin)

            # nvvideoconvert 추가 (나중에 연결됨)
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "source-nvvidconv")
            nv_bin.add(nvvidconv)

            # Ghost pad (나중에 연결됨)
            ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
            nv_bin.add_pad(ghost_pad)

            return nv_bin

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        """동적 패드 연결 콜백"""
        caps = decoder_src_pad.get_current_caps()
        struct = caps.get_structure(0)
        name = struct.get_name()

        if name.startswith("video"):
            nvvidconv = data.get_by_name("source-nvvidconv")
            if nvvidconv:
                sink_pad = nvvidconv.get_static_pad("sink")
                if not sink_pad.is_linked():
                    decoder_src_pad.link(sink_pad)

                    # Ghost pad 업데이트
                    src_pad = nvvidconv.get_static_pad("src")
                    ghost_pad = data.get_static_pad("src")
                    ghost_pad.set_target(src_pad)

    def cb_child_added(self, child_proxy, obj, name, user_data):
        """자식 요소 추가 콜백"""
        if name.find("decodebin") != -1:
            obj.connect("child-added", self.cb_child_added, user_data)

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """OSD 버퍼 프로브 - 메타데이터 처리 및 시각화"""
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

            frame_width = frame_meta.source_frame_width
            frame_height = frame_meta.source_frame_height

            # 현재 프레임의 포즈들 수집
            current_poses: List[PersonPose] = []

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.class_id == 0:  # person
                    pose = PersonPose()
                    pose.bbox = (
                        int(obj_meta.rect_params.left),
                        int(obj_meta.rect_params.top),
                        int(obj_meta.rect_params.width),
                        int(obj_meta.rect_params.height)
                    )
                    pose.score = obj_meta.confidence
                    current_poses.append(pose)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # 포즈 분석
            poses, is_fighting, fight_confidence = self.analyzer.analyze(
                current_poses, frame_width, frame_height
            )

            # 이벤트 카운트
            for pose in poses:
                if pose.is_fallen:
                    self.fall_count += 1
                    print(f"[쓰러짐 감지] 프레임 {self.frame_count}, "
                          f"신뢰도: {pose.fall_confidence:.2f}")

            if is_fighting:
                self.fight_count += 1
                print(f"[싸움 감지] 프레임 {self.frame_count}, "
                      f"신뢰도: {fight_confidence:.2f}")

            # 화면 표시 정보 추가
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1

            # FPS 및 통계 표시
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            txt_params = display_meta.text_params[0]
            txt_params.display_text = (
                f"FPS: {fps:.1f} | Falls: {self.fall_count} | "
                f"Fights: {self.fight_count} | Persons: {len(poses)}"
            )
            txt_params.x_offset = 10
            txt_params.y_offset = frame_height - 30
            txt_params.font_params.font_name = "Arial"
            txt_params.font_params.font_size = 12
            txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # 경고 표시 (쓰러짐 또는 싸움)
            has_alert = any(p.is_fallen for p in poses) or is_fighting
            if has_alert:
                alert_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                alert_meta.num_labels = 1

                alert_txt = alert_meta.text_params[0]
                if is_fighting:
                    alert_txt.display_text = f"FIGHTING DETECTED! ({fight_confidence:.2f})"
                else:
                    alert_txt.display_text = "FALL DETECTED!"

                alert_txt.x_offset = 10
                alert_txt.y_offset = 30
                alert_txt.font_params.font_name = "Arial"
                alert_txt.font_params.font_size = 20
                alert_txt.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
                alert_txt.set_bg_clr = 1
                alert_txt.text_bg_clr.set(1.0, 1.0, 1.0, 0.7)

                pyds.nvds_add_display_meta_to_frame(frame_meta, alert_meta)

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
            print("스트림 종료")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"경고: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"오류: {err}: {debug}")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"상태 변경: {old.value_nick} -> {new.value_nick}")
        return True

    def create_pipeline(self):
        """GStreamer 파이프라인 생성"""
        print("파이프라인 생성 중...")

        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            raise RuntimeError("파이프라인 생성 실패")

        # 소스 빈 생성
        source_bin = self.create_source_bin(0, self.args.source)
        self.pipeline.add(source_bin)

        # Streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", MUXER_OUTPUT_WIDTH)
        streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
        streammux.set_property("live-source", 1 if self.args.source.isdigit() else 0)
        self.pipeline.add(streammux)

        # 소스 연결
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad.link(sinkpad)

        # Primary GIE (Pose Estimation)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config", "pgie_pose_config.txt"
        )
        pgie.set_property("config-file-path", config_path)
        self.pipeline.add(pgie)

        # NvVideoConvert
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.pipeline.add(nvvidconv)

        # NvOSD
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        nvosd.set_property("process-mode", 0)  # CPU mode
        nvosd.set_property("display-text", 1)
        self.pipeline.add(nvosd)

        # Sink 설정
        if self.args.output:
            # 파일 출력
            sink = self.create_file_sink(self.args.output)
        elif self.args.rtsp:
            # RTSP 출력
            sink = self.create_rtsp_sink()
        elif self.args.no_display:
            # Fake sink (화면 없음)
            sink = Gst.ElementFactory.make("fakesink", "fake-sink")
        else:
            # EGL 디스플레이
            if os.environ.get('DISPLAY'):
                sink = Gst.ElementFactory.make("nveglglessink", "egl-sink")
            else:
                sink = Gst.ElementFactory.make("nvoverlaysink", "overlay-sink")
            sink.set_property("sync", 0)

        self.pipeline.add(sink)

        # 요소 연결
        streammux.link(pgie)
        pgie.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        # OSD 프로브 추가
        osdsinkpad = nvosd.get_static_pad("sink")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print("파이프라인 생성 완료")

    def create_file_sink(self, output_path: str):
        """파일 출력 싱크 생성"""
        queue = Gst.ElementFactory.make("queue", "queue-file")
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

        # 인코더 (H.264)
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", 4000000)

        # Parser
        parser = Gst.ElementFactory.make("h264parse", "parser")

        # Muxer
        muxer = Gst.ElementFactory.make("mp4mux", "muxer")

        # File sink
        filesink = Gst.ElementFactory.make("filesink", "filesink")
        filesink.set_property("location", output_path)
        filesink.set_property("sync", 0)

        # 빈 생성
        sink_bin = Gst.Bin.new("file-sink-bin")
        for elem in [queue, nvvidconv2, encoder, parser, muxer, filesink]:
            sink_bin.add(elem)

        queue.link(nvvidconv2)
        nvvidconv2.link(encoder)
        encoder.link(parser)
        parser.link(muxer)
        muxer.link(filesink)

        # Ghost pad
        ghost_pad = Gst.GhostPad.new("sink", queue.get_static_pad("sink"))
        sink_bin.add_pad(ghost_pad)

        return sink_bin

    def create_rtsp_sink(self):
        """RTSP 출력 싱크 생성"""
        # RTSP 서버 설정
        queue = Gst.ElementFactory.make("queue", "queue-rtsp")
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", 4000000)
        parser = Gst.ElementFactory.make("h264parse", "parser")
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")

        # UDP sink
        udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
        udpsink.set_property("host", "127.0.0.1")
        udpsink.set_property("port", self.args.rtsp_port)
        udpsink.set_property("sync", 0)

        sink_bin = Gst.Bin.new("rtsp-sink-bin")
        for elem in [queue, nvvidconv2, encoder, parser, rtppay, udpsink]:
            sink_bin.add(elem)

        queue.link(nvvidconv2)
        nvvidconv2.link(encoder)
        encoder.link(parser)
        parser.link(rtppay)
        rtppay.link(udpsink)

        ghost_pad = Gst.GhostPad.new("sink", queue.get_static_pad("sink"))
        sink_bin.add_pad(ghost_pad)

        return sink_bin

    def run(self):
        """애플리케이션 실행"""
        print("=" * 60)
        print("DeepStream Human Pose Estimation")
        print("=" * 60)
        print(f"소스: {self.args.source}")
        print(f"출력: {'화면' if not self.args.output else self.args.output}")
        print("=" * 60)

        self.create_pipeline()

        # 메인 루프
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # 파이프라인 시작
        print("파이프라인 시작...")
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        print("\n정리 중...")

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # 최종 통계
        elapsed = time.time() - self.start_time
        print("=" * 60)
        print("최종 통계:")
        print(f"  총 프레임: {self.frame_count}")
        print(f"  처리 시간: {elapsed:.2f}초")
        print(f"  평균 FPS: {self.frame_count / elapsed:.2f}" if elapsed > 0 else "")
        print(f"  쓰러짐 감지: {self.fall_count}")
        print(f"  싸움 감지: {self.fight_count}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DeepStream Human Pose Estimation Application"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='비디오 소스 (카메라 번호 또는 URI, 기본값: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 비디오 파일 경로'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='화면 표시 비활성화'
    )
    parser.add_argument(
        '--rtsp',
        action='store_true',
        help='RTSP 스트리밍 활성화'
    )
    parser.add_argument(
        '--rtsp-port',
        type=int,
        default=8554,
        help='RTSP 포트 (기본값: 8554)'
    )

    args = parser.parse_args()

    # 애플리케이션 실행
    app = DeepStreamPoseApp(args)
    app.run()


if __name__ == "__main__":
    main()
