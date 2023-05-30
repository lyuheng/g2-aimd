GCC := g++
CFLAGS += -std=c++17 -O3
INC := -I.. -I../third_party/bliss-0.73
ACC_FLAGS := -funroll-loops
LIBS := -lpthread 
BLISS_DIR = ../third_party/bliss-0.73
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

all: kmeans view_pack preprocess fennel kcore

kmeans: kmeans.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

view_pack: view_pack.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

preprocess: preprocess.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

fennel: fennel.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)
kcore: kcore.cpp
	$(GCC) $< $(CFLAGS) $(ACC_FLAGS) $(INC) -o $@ $(LIBS) $(BLISS_LIB)

clean:
	rm -rf kmeans view_pack preprocess fennel