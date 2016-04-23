gpu: BUILD/kernels.o BUILD/gpuMain.o BUILD/main.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o
	nvcc -arch=sm_52 BUILD/kernels.o BUILD/gpuMain.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o BUILD/main.o -o gpu `pkg-config opencv --cflags --libs` -lpthread
BUILD/gpuMain.o: GPU/gpuMain.cu
	nvcc -arch=sm_52 -c GPU/gpuMain.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv gpuMain.o BUILD
BUILD/kernels.o:	GPU/kernels.cu
	nvcc -arch=sm_52 -c GPU/kernels.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv kernels.o BUILD
BUILD/RegionOfInterest.o: CPU/RegionOfInterest.cpp
	g++ -c CPU/RegionOfInterest.cpp `pkg-config opencv --cflags --libs`
	mv RegionOfInterest.o BUILD
BUILD/UcharSerialCamShift.o: CPU/UcharSerialCamShift.cpp
	g++ -c CPU/UcharSerialCamShift.cpp -std=c++11 `pkg-config opencv --cflags --libs`
	mv UcharSerialCamShift.o BUILD
BUILD/main.o:	main.cpp
	g++ -c main.cpp -std=c++11 `pkg-config opencv --cflags --libs`
	mv main.o BUILD
clean:
	rm BUILD/*.o gpu out.mov 
run:
	./gpu in.mov windows

