gpu: BUILD/gpuMerge.o BUILD/gpuMain.o BUILD/main.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o BUILD/timing.o
	nvcc -arch=sm_30 BUILD/gpuMerge.o BUILD/gpuMain.o BUILD/timing.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o  BUILD/main.o -o gpu `pkg-config opencv --cflags --libs`
BUILD/gpuMain.o: GPU/gpuMain.cu
	nvcc -arch=sm_30 -c GPU/gpuMain.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv gpuMain.o BUILD
BUILD/gpuMerge.o:	GPU/gpuMerge.cu
	nvcc -arch=sm_30 -c GPU/gpuMerge.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv gpuMerge.o BUILD
BUILD/timing.o: GPU/timing.c
	gcc -c GPU/timing.c
	mv timing.o BUILD
BUILD/RegionOfInterest.o: CPU/RegionOfInterest.cpp
	g++ -c CPU/RegionOfInterest.cpp `pkg-config opencv --cflags --libs`
	mv RegionOfInterest.o BUILD
BUILD/UcharSerialCamShift.o: CPU/UcharSerialCamShift.cpp
	g++ -c CPU/UcharSerialCamShift.cpp `pkg-config opencv --cflags --libs`
	mv UcharSerialCamShift.o BUILD
#cpuMain.o:	CPU/cpuMain.cpp
#	g++ -c CPU/cpuMain.cpp `pkg-config opencv --cflags --libs`
BUILD/main.o:	main.cpp
	g++ -c main.cpp -std=c++11 `pkg-config opencv --cflags --libs`
	mv main.o BUILD
clean:
	rm BUILD/*.o gpu out.mov *.txt
run:
	./gpu in.mov windows

