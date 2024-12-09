
# Radix Sort Benchmark: CPU vs GPU

Este proyecto implementa y compara el rendimiento del algoritmo **Radix Sort** en dos plataformas: **CPU** (usando OpenMP) y **GPU** (usando CUDA). Además, genera gráficos para analizar los resultados del rendimiento.

## Estructura del Proyecto

- **`radix_sort.cu`**: Código principal que implementa Radix Sort tanto en CPU como en GPU.
- **`Makefile`**: Archivo Make para compilar y ejecutar el proyecto.
- **`Graficos/output.txt`**: Archivo generado tras ejecutar el programa, donde se almacenan los tiempos de ejecución de las iteraciones.
- **`radix_sort_benchmark.png`**: Imagen del gráfico generado tras ejecutar el script Python.

---

## Requisitos

### Software
1. **NVIDIA CUDA Toolkit**: Para compilar y ejecutar el código en GPU.
2. **GCC**: Para compilar el código en CPU con soporte OpenMP.
3. **GNU Make**: Para gestionar la compilación.
4. **CMake** (opcional): En caso de personalizar el proceso de compilación.

### Hardware
- GPU compatible con **CUDA**.
- CPU multinúcleo para aprovechar **OpenMP**.

---

## Instalación y Compilación

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

## Ejecución

### Modo Benchmark
El programa compara el rendimiento de Radix Sort en CPU y GPU. Usa los siguientes comandos:

#### Ejecutar en CPU
```bash
./Ejecutables/prog <n> <num_hilos>
```
- `<n>`: Tamaño del arreglo (e.g., `1000000`).
- `<num_hilos>`: Número de hilos para OpenMP.

## Ejemplo de Uso

### Ejecución de Benchmark
1. Compila el proyecto:
   ```bash
   make
   ```

2. Ejecuta un benchmark para comparar CPU y GPU:
   ```bash
   ./Ejecutables/prog 1000000 8
   ```
---

## Resultados Esperados

- **`output.txt`** contendrá los tiempos de ejecución de las iteraciones para CPU y GPU.
- **`radix_sort_benchmark.png`** mostrará un gráfico comparativo, donde se observará que la GPU suele ser significativamente más rápida para tamaños grandes de datos.

---

## Notas Adicionales

### Problemas Comunes
- **CUDA Error**: Asegúrate de tener un controlador NVIDIA y CUDA Toolkit instalado.
- **Errores de Compilación**: Confirma que `nvcc` está configurado en el PATH del sistema.

---

## Licencia

Este proyecto está bajo la **Licencia MIT**, lo que significa que puedes usar, modificar y distribuir el código libremente, siempre y cuando des crédito al autor original.
