
# Radix Sort Benchmark: CPU vs GPU

Este proyecto implementa y compara el rendimiento del algoritmo **Radix Sort** en dos plataformas: **CPU** (usando OpenMP) y **GPU** (usando CUDA). Ademas, genera graficos para analizar los resultados del rendimiento.

## Estructura del Proyecto

- **`radix_sort.cu`**: Codigo principal que implementa Radix Sort tanto en CPU como en GPU.
- **`Gen_Graficos.py`**: Script Python que genera un grafico comparativo del tiempo de ejecucion entre CPU y GPU.
- **`Makefile`**: Archivo Make para compilar y ejecutar el proyecto.
- **`Graficos/output.txt`**: Archivo generado tras ejecutar el programa, donde se almacenan los tiempos de ejecucion de las iteraciones.
- **`radix_sort_benchmark.png`**: Imagen del grafico generado tras ejecutar el script Python.

---

## Requisitos

### Software
1. **NVIDIA CUDA Toolkit**: Para compilar y ejecutar el codigo en GPU.
2. **Python 3**: Para generar los graficos (requerimientos adicionales: `matplotlib`).
3. **GCC**: Para compilar el codigo en CPU con soporte OpenMP.
4. **GNU Make**: Para gestionar la compilacion.
5. **CMake** (opcional): En caso de personalizar el proceso de compilacion.

### Hardware
- GPU compatible con **CUDA**.
- CPU multinucleo para aprovechar **OpenMP**.

---

## Instalacion y Compilacion

1. **Clona el repositorio** (o copia los archivos necesarios en un directorio):
   ```bash
   git clone <repositorio-url>
   cd <directorio-del-proyecto>
   ```

2. **Compila el proyecto** usando el archivo `Makefile`:
   ```bash
   make
   ```

3. (Opcional) Limpia los archivos generados:
   ```bash
   make clean
   ```

---

## Ejecucion

### Modo Benchmark
El programa compara el rendimiento de Radix Sort en CPU y GPU. Usa los siguientes comandos:

#### Ejecutar en CPU
```bash
./Ejecutables/prog <n> <num_hilos>
```
- `<n>`: Tamaño del arreglo (e.g., `1000000`).
- `<num_hilos>`: Numero de hilos para OpenMP.

#### Ejecutar en GPU
```bash
./Ejecutables/prog <n> 0
```
- `<n>`: Tamaño del arreglo (e.g., `1000000`).

### Generar Graficos
1. Tras ejecutar el programa, el archivo `Graficos/output.txt` contendra los resultados.
2. Usa el script Python para generar los graficos:
   ```bash
   python3 Gen_Graficos.py
   ```
3. El grafico se guardara como `radix_sort_benchmark.png`.

---

## Ejemplo de Uso

### Ejecucion de Benchmark
1. Compila el proyecto:
   ```bash
   make
   ```

2. Ejecuta un benchmark para comparar CPU y GPU:
   ```bash
   ./Ejecutables/prog 1000000 8
   ```

3. Genera el grafico de los resultados:
   ```bash
   python3 Gen_Graficos.py
   ```

---

## Resultados Esperados

- **`output.txt`** contendra los tiempos de ejecucion de las iteraciones para CPU y GPU.
- **`radix_sort_benchmark.png`** mostrara un grafico comparativo, donde se observara que la GPU suele ser significativamente mas rapida para tamaños grandes de datos.

---

## Notas Adicionales

### Problemas Comunes
- **CUDA Error**: Asegurate de tener un controlador NVIDIA y CUDA Toolkit instalado.
- **Errores de Compilacion**: Confirma que `nvcc` esta configurado en el PATH del sistema.

### Requerimientos de Python
Instala `matplotlib` si no esta disponible:
```bash
pip install matplotlib
```

---

## Licencia

Este proyecto esta bajo la **Licencia MIT**, lo que significa que puedes usar, modificar y distribuir el codigo libremente, siempre y cuando des credito al autor original.