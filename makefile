# Nombre del compilador
CXX = g++

# Opciones del compilador
CXXFLAGS = -std=c++11 -Wall -Wextra

# Nombre del ejecutable
TARGET = Ejecutables/main

# Archivos fuente
SRC = Main.cpp

# Archivos objeto
OBJ = $(patsubst %.cpp,Ejecutables/%.o,$(SRC))

# Regla predeterminada
all: $(TARGET)

# Regla para crear el ejecutable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# Regla para compilar los archivos fuente a objeto
Ejecutables/%.o: %.cpp
	@mkdir -p Ejecutables
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(OBJ) $(TARGET)

# Regla de limpieza completa
clean-all: clean
	rm -rf Ejecutables/*.o
