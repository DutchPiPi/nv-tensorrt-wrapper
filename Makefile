CUDA_PATH ?= /usr/local/cuda
TRT_PATH ?= /usr/local/tensorrt7
ARCH ?= sm_75
BUILD_DIR = build
PREFIX ?= /usr/local
LIBDIR = lib
INSTALL = install

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -DNDEBUG -std=c++17
INCLUDES := -I$(CUDA_PATH)/include -isystem $(TRT_PATH)/include  -I../libavfilter/dnn/
LDFLAGS := -L/usr/local/cuda/lib64 -L$(TRT_PATH)/lib
LDFLAGS += -lnvinfer -lcudart -lz -ldl

OBJ = $(shell find $(BUILD_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

SO = $(addprefix $(BUILD_DIR)/, libnvtensorrt.so)

all: $(SO)

$(BUILD_DIR)/libnvtensorrt.so: $(addprefix $(BUILD_DIR)/, trt_class_wrapper.o)

-include $(DEP)

clean:
	rm -rf $(SO) $(OBJ) $(DEP)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=$(ARCH) -o $@ -c $<

$(SO):
	$(NVCC) $(CCFLAGS) -shared -Xcompiler -fPIC -o $@ $+ $(LDFLAGS)

install: all
	$(INSTALL) -m 0755 -d '../libavfilter/dnn/'
	$(INSTALL) -m 0644 trt_class_wrapper.h '../libavfilter/dnn/'
	$(INSTALL) -m 0755 -d '$(PREFIX)/$(LIBDIR)'
	$(INSTALL) -m 0644 $(BUILD_DIR)/libnvtensorrt.so '$(PREFIX)/$(LIBDIR)'

uninstall:
	rm -rf '$(PREFIX)/include/ffnvtensorrt' '$(PREFIX)/$(LIBDIR)/libnvtensorrt.so'

.PHONY: all install uninstall

