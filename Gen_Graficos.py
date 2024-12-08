import matplotlib.pyplot as plt

# Leer los datos del archivo
iterations = []
cpu_times = []
gpu_times = []

with open("/Graficos/output.txt", "r") as file:
    for line in file:
        if "Iteración" in line:
            parts = line.split()
            iteration = int(parts[1])
            time = float(parts[2])
            if iteration not in iterations:
                iterations.append(iteration)
                cpu_times.append(time)
            else:
                gpu_times.append(time)

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Gráfico de CPU
plt.plot(iterations, cpu_times, label="CPU", color='b', marker='o')

# Gráfico de GPU
plt.plot(iterations, gpu_times, label="GPU", color='r', marker='x')

# Configurar el gráfico
plt.xlabel("Iteración")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Benchmark de Radix Sort: CPU vs GPU")
plt.legend()

# Guardar la imagen
plt.savefig("radix_sort_benchmark.png")
plt.show()
