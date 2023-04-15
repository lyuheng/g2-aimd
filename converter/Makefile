GCC := g++
CFLAGS += -std=c++17 -O3
INC := -I.. -I../third_party/bliss-0.73
ACC_FLAGS := -funroll-loops
LIBS := -lpthread 
BLISS_DIR = ../third_party/bliss-0.73
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

all: toGAMMA toGSI toPangolin

toGAMMA: toGAMMA.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

toGSI: toGSI.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

toPangolin: toPangolin.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

clean:
	rm -rf toGAMMA toGSI toPangolin