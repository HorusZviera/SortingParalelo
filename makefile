# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2 -fopenmp

# Target executable
TARGET = sorting_paralelo

# Source files
SRCS = CPU.cu GPU.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean