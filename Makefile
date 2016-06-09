gpu: BUILD/dynamicCamShift.o BUILD/deviceProperties.o BUILD/gpuMain.o BUILD/main.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o
	nvcc -arch=sm_52 BUILD/dynamicCamShift.o BUILD/deviceProperties.o BUILD/gpuMain.o BUILD/RegionOfInterest.o BUILD/UcharSerialCamShift.o BUILD/main.o -o BUILD/gpuMeanShift `pkg-config opencv --cflags --libs` -lpthread
BUILD/gpuMain.o: GPU/gpuMain.cu
	nvcc -arch=sm_52 -c GPU/gpuMain.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv gpuMain.o BUILD
BUILD/dynamicCamShift.o: GPU/dynamicCamShift.cu
	nvcc -arch=sm_52 -rdc=true -lcudadevrt -c GPU/dynamicCamShift.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv dynamicCamShift.o BUILD
BUILD/deviceProperties.o:	GPU/deviceProperties.cu
	nvcc -arch=sm_52 -c GPU/deviceProperties.cu -I/usr/local.cuda-7.0/samples/common -I../../common/inc
	mv deviceProperties.o BUILD
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
	rm BUILD/* out.mov 
run:
	./BUILD/gpuMeanShift INPUT/in4.mov INPUT/windows.in1.test1

