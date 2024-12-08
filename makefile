NVCC = nvcc
CXXFLAGS = -Xcompiler -fopenmp
LDFLAGS = -lgomp
TARGET_DIR = Ejecutables
TARGET = $(TARGET_DIR)/prog
SRC = radix_sort.cu

all: $(TARGET_DIR) $(TARGET)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run_cpu:
	./$(TARGET) $((2**26)) 0 8

run_gpu:
	./$(TARGET) $((2**26)) 1 0
