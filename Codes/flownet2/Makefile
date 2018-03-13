# Makefile

TF_INC = `python -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`

ifndef CUDA_HOME
    CUDA_HOME := /usr/local/cuda
endif

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -I"$(CUDA_HOME)/include" -DGOOGLE_CUDA=1
GPUCFLAGS = -c
LFLAGS    = -pthread -shared -fPIC
GPULFLAGS = -x cu -Xcompiler -fPIC
CGPUFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart

OUT_DIR   = src/ops/build
PREPROCESSING_SRC = "src/ops/preprocessing/preprocessing.cc" "src/ops/preprocessing/kernels/flow_augmentation.cc" "src/ops/preprocessing/kernels/augmentation_base.cc" "src/ops/preprocessing/kernels/data_augmentation.cc"
GPU_SRC_DATA_AUG  	= src/ops/preprocessing/kernels/data_augmentation.cu.cc
GPU_SRC_FLOW     	= src/ops/preprocessing/kernels/flow_augmentation_gpu.cu.cc
GPU_PROD_DATA_AUG 	= $(OUT_DIR)/data_augmentation.o
GPU_PROD_FLOW    	= $(OUT_DIR)/flow_augmentation_gpu.o
PREPROCESSING_PROD	= $(OUT_DIR)/preprocessing.so

DOWNSAMPLE_SRC = "src/ops/downsample/downsample_kernel.cc" "src/ops/downsample/downsample_op.cc"
GPU_SRC_DOWNSAMPLE  = src/ops/downsample/downsample_kernel_gpu.cu.cc
GPU_PROD_DOWNSAMPLE = $(OUT_DIR)/downsample_kernel_gpu.o
DOWNSAMPLE_PROD 	= $(OUT_DIR)/downsample.so

CORRELATION_SRC = "src/ops/correlation/correlation_kernel.cc" "src/ops/correlation/correlation_grad_kernel.cc" "src/ops/correlation/correlation_op.cc"
GPU_SRC_CORRELATION  = src/ops/correlation/correlation_kernel.cu.cc
GPU_SRC_CORRELATION_GRAD  = src/ops/correlation/correlation_grad_kernel.cu.cc
GPU_SRC_PAD = src/ops/correlation/pad.cu.cc
GPU_PROD_CORRELATION = $(OUT_DIR)/correlation_kernel_gpu.o
GPU_PROD_CORRELATION_GRAD = $(OUT_DIR)/correlation_grad_kernel_gpu.o
GPU_PROD_PAD = $(OUT_DIR)/correlation_pad_gpu.o
CORRELATION_PROD 	= $(OUT_DIR)/correlation.so

FLOWWARP_SRC = "src/ops/flow_warp/flow_warp_op.cc" "src/ops/flow_warp/flow_warp.cc" "src/ops/flow_warp/flow_warp_grad.cc"
GPU_SRC_FLOWWARP = "src/ops/flow_warp/flow_warp.cu.cc"
GPU_SRC_FLOWWARP_GRAD = "src/ops/flow_warp/flow_warp_grad.cu.cc"
GPU_PROD_FLOWWARP = "$(OUT_DIR)/flow_warp_gpu.o"
GPU_PROD_FLOWWARP_GRAD = "$(OUT_DIR)/flow_warp_grad_gpu.o"
FLOWWARP_PROD = "$(OUT_DIR)/flow_warp.so"

ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif
ifeq ($(detected_OS),Darwin)  # Mac OS X
	CGPUFLAGS += -undefined dynamic_lookup
endif
ifeq ($(detected_OS),Linux)
	CFLAGS += -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -D_GLIBCXX_USE_CXX11_ABI=0
endif

all: preprocessing downsample correlation flowwarp

preprocessing:
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_DATA_AUG) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_DATA_AUG)
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_FLOW) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_FLOW)
	$(CXX) -g $(CFLAGS)  $(PREPROCESSING_SRC) $(GPU_PROD_DATA_AUG) $(GPU_PROD_FLOW) $(LFLAGS) $(CGPUFLAGS) -o $(PREPROCESSING_PROD)

downsample:
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_DOWNSAMPLE) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_DOWNSAMPLE)
	$(CXX) -g $(CFLAGS)  $(DOWNSAMPLE_SRC) $(GPU_PROD_DOWNSAMPLE) $(LFLAGS) $(CGPUFLAGS) -o $(DOWNSAMPLE_PROD)

correlation:
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_CORRELATION) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_CORRELATION)
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_CORRELATION_GRAD) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_CORRELATION_GRAD)
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_PAD) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_PAD)
	$(CXX) -g $(CFLAGS)  $(CORRELATION_SRC) $(GPU_PROD_CORRELATION) $(GPU_PROD_CORRELATION_GRAD) $(GPU_PROD_PAD) $(LFLAGS) $(CGPUFLAGS) -o $(CORRELATION_PROD)

flowwarp:
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_FLOWWARP) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_FLOWWARP)
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_FLOWWARP_GRAD) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_FLOWWARP_GRAD)
	$(CXX) -g $(CFLAGS)  $(FLOWWARP_SRC) $(GPU_PROD_FLOWWARP) $(GPU_PROD_FLOWWARP_GRAD) $(LFLAGS) $(CGPUFLAGS) -o $(FLOWWARP_PROD)

clean:
	rm -f $(PREPROCESSING_PROD) $(GPU_PROD_FLOW) $(GPU_PROD_DATA_AUG) $(DOWNSAMPLE_PROD) $(GPU_PROD_DOWNSAMPLE)
