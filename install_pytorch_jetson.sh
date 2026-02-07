#!/bin/bash
# Jetson AGX Orin용 PyTorch 설치 스크립트
# JetPack 5.x (L4T R35.x) / CUDA 11.4 / Python 3.8

set -e

echo "=========================================="
echo "Jetson AGX Orin PyTorch 설치 스크립트"
echo "JetPack: R35.6.3, CUDA: 11.4, Python: 3.8"
echo "=========================================="

# 1. 시스템 의존성 설치
echo ""
echo "[1/5] 시스템 의존성 설치 중..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# 2. pip 업그레이드
echo ""
echo "[2/5] pip 업그레이드 중..."
pip3 install --upgrade pip

# 3. PyTorch 설치 (NVIDIA 공식 휠)
echo ""
echo "[3/5] PyTorch 2.1.0 설치 중... (JetPack 5.x용)"
# NVIDIA 공식 PyTorch 휠 URL (JetPack 5.x / CUDA 11.4 / Python 3.8)
# https://developer.nvidia.com/embedded/downloads 에서 확인 가능

# PyTorch 2.1.0 for JetPack 5.1.2+ (CUDA 11.4)
pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# 4. torchvision 설치 (소스에서 빌드)
echo ""
echo "[4/5] torchvision 0.16.0 설치 중..."

# 먼저 빌드 의존성 설치
sudo apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

# torchvision 소스에서 빌드
pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.0a0+781e5a3-cp38-cp38-linux_aarch64.whl \
    2>/dev/null || {
    echo "사전 빌드된 torchvision을 찾을 수 없습니다. 소스에서 빌드합니다..."
    cd /tmp
    git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.16.0
    pip3 install -e .
    cd ..
    rm -rf torchvision
}

# 5. 설치 확인
echo ""
echo "[5/5] 설치 확인 중..."
python3 -c "
import torch
print('========================================')
print('PyTorch 설치 완료!')
print('========================================')
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 이름: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('========================================')
"

# torchvision 확인
python3 -c "
import torchvision
print(f'torchvision 버전: {torchvision.__version__}')
" 2>/dev/null || echo "torchvision 설치 필요"

echo ""
echo "설치가 완료되었습니다!"
echo "이제 다음 명령으로 YOLO를 GPU에서 실행할 수 있습니다:"
echo "  python3 main.py --web --use-yolo --device cuda"
