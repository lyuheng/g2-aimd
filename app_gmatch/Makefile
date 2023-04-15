NVCC := nvcc
INC := -I.. -I../third_party/MPMCQueue/include -I../third_party/bliss-0.73
CFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp  -lgomp -lpthread -DOPENMP

BLISS_DIR = ../third_party/bliss-0.73
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

ARCH = -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -Xptxas -v -maxrregcount 64 

all: clean run

run: main.cu
	$(NVCC) $< $(CFLAGS) ${ARCH} $(INC) $(BLISS_LIB) -o $@

clean:
	rm -rf run