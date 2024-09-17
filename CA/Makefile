CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`

all: main

main: main.o cuda1.o cuda2.o
	$(CC) main.o cuda1.o cuda2.o -o main $(LINKER_DIRS) $(OCELOT) -lrt

main.o: main.cu
	$(NVCC) main.cu -c -I . 

cuda2.o: cuda2.cu
	$(NVCC) -c cuda2.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

cuda1.o: cuda1.cu
	$(NVCC) -c cuda1.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

clean:
	rm -f main.o cuda1.o cuda2.o main kernel-times.json
