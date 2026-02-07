# DeepStream Pose Estimation - Makefile
# Jetson 및 x86 플랫폼 지원

# 플랫폼 감지
PLATFORM ?= $(shell uname -m)

# DeepStream 경로
DS_VERSION ?= 6.3
DS_PATH ?= /opt/nvidia/deepstream/deepstream-$(DS_VERSION)

# CUDA 경로
ifeq ($(PLATFORM), aarch64)
    # Jetson
    CUDA_PATH ?= /usr/local/cuda
    NVCC_FLAGS = -gencode arch=compute_72,code=sm_72 \
                 -gencode arch=compute_87,code=sm_87
else
    # x86
    CUDA_PATH ?= /usr/local/cuda
    NVCC_FLAGS = -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86
endif

# 컴파일러 설정
CXX = g++
NVCC = $(CUDA_PATH)/bin/nvcc

# 디렉토리
SRC_DIR = src
LIB_DIR = lib
CONFIG_DIR = config
MODEL_DIR = models

# 소스 파일
PARSER_SRC = $(SRC_DIR)/nvdsinfer_pose_parser.cpp

# 출력 라이브러리
PARSER_LIB = $(LIB_DIR)/libnvds_infercustomparser_pose.so

# 컴파일 플래그
CXXFLAGS = -std=c++14 -fPIC -Wall -Wno-deprecated-declarations
CXXFLAGS += -I$(DS_PATH)/sources/includes
CXXFLAGS += -I$(CUDA_PATH)/include
CXXFLAGS += -I$(SRC_DIR)

# 최적화 플래그
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O3
endif

# 링커 플래그
LDFLAGS = -shared
LDFLAGS += -L$(DS_PATH)/lib
LDFLAGS += -L$(CUDA_PATH)/lib64
LDFLAGS += -lnvinfer -lnvinfer_plugin
LDFLAGS += -lcudart

# 기본 타겟
.PHONY: all clean install test

all: $(PARSER_LIB)

# 디렉토리 생성
$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

$(MODEL_DIR):
	@mkdir -p $(MODEL_DIR)

# 파서 라이브러리 빌드
$(PARSER_LIB): $(PARSER_SRC) | $(LIB_DIR)
	@echo "===== Building Pose Parser Library ====="
	@echo "Platform: $(PLATFORM)"
	@echo "DeepStream: $(DS_PATH)"
	@echo "CUDA: $(CUDA_PATH)"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "===== Build Complete: $@ ====="

# 설치
install: $(PARSER_LIB)
	@echo "Installing to $(DS_PATH)/lib/"
	sudo cp $(PARSER_LIB) $(DS_PATH)/lib/
	@echo "Installation complete!"

# 모델 다운로드
download-model: | $(MODEL_DIR)
	@echo "Downloading trt_pose model..."
	python3 download_model.py --model resnet18_baseline --output-dir $(MODEL_DIR)

# 정리
clean:
	rm -rf $(LIB_DIR)
	rm -f *.engine

# 전체 정리
distclean: clean
	rm -rf $(MODEL_DIR)

# 테스트 실행
test: $(PARSER_LIB)
	@echo "Running test pipeline..."
	python3 deepstream_pose_app.py --source 0

# 도움말
help:
	@echo "DeepStream Pose Estimation Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build parser library (default)"
	@echo "  install        - Install library to DeepStream"
	@echo "  download-model - Download trt_pose model"
	@echo "  clean          - Remove build artifacts"
	@echo "  distclean      - Remove all generated files"
	@echo "  test           - Run test application"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  DS_VERSION     - DeepStream version (default: 6.3)"
	@echo "  DS_PATH        - DeepStream path"
	@echo "  CUDA_PATH      - CUDA path"
	@echo "  DEBUG          - Enable debug build (DEBUG=1)"
	@echo "  PLATFORM       - Target platform (aarch64/x86_64)"
