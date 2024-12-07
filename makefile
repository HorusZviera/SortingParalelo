NVCC = nvcc
CXXFLAGS = -Xcompiler -fopenmp
LDFLAGS = -lgomp
TARGET = prog
SRC = radix_sort.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run_cpu:
	./$(TARGET) $((2**26)) 0 8

run_gpu:
	./$(TARGET) $((2**26)) 1 0
