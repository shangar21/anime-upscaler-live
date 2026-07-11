################################################################################
# Adapted from NVIDIA's reference gst-nvdsvideotemplate Makefile.
# Build with:
#   CUDA_VER=13.2 make
# (match CUDA_VER to whatever `nvcc --version` / your CUDA install reports)
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set, e.g. CUDA_VER=13.2 make")
endif

CXX:= g++
NVCC:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc

SRCS:= customlib_impl.cpp sr_trt_engine.cpp sr_pipeline.cpp
CU_SRCS:= pack_rgba_kernel.cu tile_kernels.cu

INCS:= $(wildcard *.hpp) $(wildcard *.h)
LIB:=libcustom_videoimpl.so

NVDS_VERSION:=9.0
DS_ROOT:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)

CFLAGS+= -O2 -fPIC -std=c++17 -DDS_VERSION=\"9.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I $(DS_ROOT)/sources/includes \
	 -I $(DS_ROOT)/sources/gst-plugins/gst-nvdsvideotemplate/includes

GST_INSTALL_DIR?=$(DS_ROOT)/lib/gst-plugins/
LIB_INSTALL_DIR?=$(DS_ROOT)/lib/

LIBS += -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl -lpthread \
	-lnppc -lnppial -lnppicc -lnppidei -lnppitc \
	-lnvinfer \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)
CU_OBJS:= $(CU_SRCS:.cu=.o)
PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ -Xcompiler -fPIC -std=c++17 -arch=sm_86 -I /usr/local/cuda-$(CUDA_VER)/include $<

$(LIB): $(OBJS) $(CU_OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(CU_OBJS) $(LIBS)

install: $(LIB)
	cp -v $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(CU_OBJS) $(LIB)
