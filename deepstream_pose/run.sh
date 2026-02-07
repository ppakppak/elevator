#!/bin/bash
# DeepStream Pose Estimation 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 환경 변수 설정
export GST_DEBUG=2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}"
    echo "=============================================="
    echo "  DeepStream Human Pose Estimation"
    echo "=============================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the custom parser library"
    echo "  download    - Download and convert the model"
    echo "  run         - Run the application"
    echo "  test        - Run a quick test"
    echo "  clean       - Clean build artifacts"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build parser library"
    echo "  $0 download                 # Download model"
    echo "  $0 run                      # Run with default camera"
    echo "  $0 run --source 0           # Run with USB camera"
    echo "  $0 run --source video.mp4   # Run with video file"
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # DeepStream 확인
    if [ ! -d "/opt/nvidia/deepstream/deepstream" ]; then
        echo -e "${RED}Error: DeepStream SDK not found${NC}"
        echo "Please install DeepStream SDK first"
        exit 1
    fi
    echo "  ✓ DeepStream SDK found"

    # CUDA 확인
    if [ ! -d "/usr/local/cuda" ]; then
        echo -e "${RED}Error: CUDA not found${NC}"
        exit 1
    fi
    echo "  ✓ CUDA found"

    # Python 확인
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 not found${NC}"
        exit 1
    fi
    echo "  ✓ Python3 found"

    # GStreamer 확인
    if ! gst-inspect-1.0 nvinfer &> /dev/null; then
        echo -e "${RED}Error: GStreamer nvinfer plugin not found${NC}"
        exit 1
    fi
    echo "  ✓ GStreamer plugins found"

    echo -e "${GREEN}All dependencies OK${NC}"
    echo ""
}

build_parser() {
    echo -e "${YELLOW}Building custom parser library...${NC}"
    make clean
    make

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build successful!${NC}"
    else
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
}

download_model() {
    echo -e "${YELLOW}Downloading and converting model...${NC}"
    python3 download_model.py --model resnet18_baseline

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model ready!${NC}"
    else
        echo -e "${RED}Model download/conversion failed!${NC}"
        exit 1
    fi
}

run_app() {
    echo -e "${YELLOW}Starting application...${NC}"

    # 파서 라이브러리 확인
    if [ ! -f "lib/libnvds_infercustomparser_pose.so" ]; then
        echo -e "${YELLOW}Parser library not found, building...${NC}"
        build_parser
    fi

    # 모델 확인
    if [ ! -f "models/resnet18_baseline_224x224.onnx" ]; then
        echo -e "${YELLOW}Model not found, downloading...${NC}"
        download_model
    fi

    # 시스템 Python 사용 (gi, pyds 모듈 필요)
    /usr/bin/python3 deepstream_pose_app.py "$@"
}

# 메인 로직
print_header

case "$1" in
    build)
        check_dependencies
        build_parser
        ;;
    download)
        check_dependencies
        download_model
        ;;
    run)
        check_dependencies
        shift
        run_app "$@"
        ;;
    test)
        check_dependencies
        build_parser
        download_model
        run_app --source 0 --no-display
        ;;
    clean)
        make clean
        echo -e "${GREEN}Cleaned!${NC}"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        if [ -z "$1" ]; then
            # 기본: 실행
            check_dependencies
            run_app
        else
            echo -e "${RED}Unknown command: $1${NC}"
            print_usage
            exit 1
        fi
        ;;
esac
