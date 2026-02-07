#!/usr/bin/env python3
"""
trt_pose 모델 다운로드 및 ONNX 변환 스크립트
DeepStream에서 사용하기 위한 ONNX 모델을 생성합니다.
"""

import os
import sys
import argparse
import subprocess
import urllib.request
from pathlib import Path


# 모델 URL (NVIDIA trt_pose GitHub에서 제공하는 사전 훈련 모델)
MODEL_URLS = {
    "resnet18_baseline": {
        "url": "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth",
        "input_size": (224, 224),
        "backbone": "resnet18"
    },
    "densenet121_baseline": {
        "url": "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/densenet121_baseline_att_256x256_B_epoch_160.pth",
        "input_size": (256, 256),
        "backbone": "densenet121"
    }
}

# COCO 키포인트 정의 (18개 관절)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
]

# 스켈레톤 연결 정의
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [0, 17], [17, 5], [17, 6]
]


def download_model(model_name: str, output_dir: str) -> str:
    """모델 다운로드"""
    if model_name not in MODEL_URLS:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 사용 가능: {list(MODEL_URLS.keys())}")

    model_info = MODEL_URLS[model_name]
    url = model_info["url"]
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"모델이 이미 존재합니다: {output_path}")
        return output_path

    print(f"모델 다운로드 중: {url}")
    print(f"저장 위치: {output_path}")

    os.makedirs(output_dir, exist_ok=True)

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r다운로드 진행률: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print("\n다운로드 완료!")

    return output_path


def convert_to_onnx(model_name: str, pth_path: str, output_dir: str) -> str:
    """PyTorch 모델을 ONNX로 변환"""
    try:
        import torch
        import torchvision
    except ImportError:
        print("오류: torch 및 torchvision이 필요합니다.")
        print("설치: pip install torch torchvision")
        sys.exit(1)

    model_info = MODEL_URLS[model_name]
    input_size = model_info["input_size"]
    backbone = model_info["backbone"]

    onnx_filename = f"{model_name}_{input_size[0]}x{input_size[1]}.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    if os.path.exists(onnx_path):
        print(f"ONNX 모델이 이미 존재합니다: {onnx_path}")
        return onnx_path

    print(f"ONNX 변환 중...")
    print(f"입력 크기: {input_size}")
    print(f"백본: {backbone}")

    try:
        # trt_pose 패키지가 설치되어 있는지 확인
        import trt_pose.coco
        import trt_pose.models

        num_parts = len(COCO_KEYPOINTS)
        num_links = len(SKELETON)

        # 모델 생성
        if backbone == "resnet18":
            model = trt_pose.models.resnet18_baseline_att(
                num_parts, 2 * num_links
            ).eval()
        elif backbone == "densenet121":
            model = trt_pose.models.densenet121_baseline_att(
                num_parts, 2 * num_links
            ).eval()
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")

        # 가중치 로드
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))

        # CUDA로 이동 (가능한 경우)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

        # ONNX 내보내기
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['cmap', 'paf'],  # heatmap과 part affinity field
            dynamic_axes={
                'input': {0: 'batch_size'},
                'cmap': {0: 'batch_size'},
                'paf': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True
        )

        print(f"ONNX 변환 완료: {onnx_path}")
        return onnx_path

    except ImportError:
        print("경고: trt_pose가 설치되어 있지 않습니다.")
        print("trt_pose 설치 방법:")
        print("  git clone https://github.com/NVIDIA-AI-IOT/trt_pose")
        print("  cd trt_pose")
        print("  python setup.py install")
        print("")
        print("대안: 사전 변환된 ONNX 모델을 사용하거나 수동으로 변환하세요.")
        return None


def create_human_pose_json(output_dir: str) -> str:
    """human_pose.json 생성 (키포인트 및 스켈레톤 정의)"""
    import json

    pose_config = {
        "keypoints": COCO_KEYPOINTS,
        "skeleton": SKELETON
    }

    json_path = os.path.join(output_dir, "human_pose.json")
    with open(json_path, 'w') as f:
        json.dump(pose_config, f, indent=2)

    print(f"human_pose.json 생성: {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="trt_pose 모델 다운로드 및 ONNX 변환")
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18_baseline',
        choices=list(MODEL_URLS.keys()),
        help='사용할 모델 (기본값: resnet18_baseline)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='출력 디렉토리 (기본값: models)'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='다운로드만 수행 (ONNX 변환 생략)'
    )

    args = parser.parse_args()

    # 현재 스크립트 디렉토리 기준으로 출력 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("trt_pose 모델 다운로드 및 ONNX 변환")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60)

    # 1. 모델 다운로드
    pth_path = download_model(args.model, output_dir)

    # 2. ONNX 변환
    if not args.download_only:
        onnx_path = convert_to_onnx(args.model, pth_path, output_dir)
        if onnx_path:
            print(f"\nONNX 모델 경로: {onnx_path}")

    # 3. human_pose.json 생성
    json_path = create_human_pose_json(output_dir)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n다음 단계:")
    print(f"1. DeepStream 설정 파일에서 ONNX 모델 경로 지정")
    print(f"2. TensorRT 엔진 생성 (첫 실행 시 자동)")


if __name__ == "__main__":
    main()
