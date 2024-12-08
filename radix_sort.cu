#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iomanip>
#include <fstream>

using namespace std;

// Radix Sort en CPU usando OpenMP
void counting_sort_cpu(vector<int>& vec, int exp, int nt) {
    int n = vec.size();
    vector<int> output(n);
    vector<int> count(10, 0);

    // Paso 1: Contar las ocurrencias de dígitos
    #pragma omp parallel num_threads(nt)
    {
        vector<int> local_count(10, 0);

        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            local_count[(vec[i] / exp) % 10]++;
        }

        #pragma omp critical
        for (int i = 0; i < 10; i++) {
            count[i] += local_count[i];
        }
    }

    // Paso 2: Sumar acumulativamente
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Paso 3: Construir el arreglo de salida
    #pragma omp parallel for num_threads(nt)
    for (int i = n - 1; i >= 0; i--) {
        int digit = (vec[i] / exp) % 10;
        int pos = count[digit] - 1;
        output[pos] = vec[i];
        count[digit]--;
    }

    // Paso 4: Copiar al arreglo original
    vec = output;
}

void radix_sort_cpu(vector<int>& vec, int nt) {
    int max_val = *max_element(vec.begin(), vec.end());

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_cpu(vec, exp, nt);
    }
}

extern "C" void ordenar_radix_sort_cpu_wrapper(int* arr, int size, int nt) {
    vector<int> vec(arr, arr + size);
    radix_sort_cpu(vec, nt);
    copy(vec.begin(), vec.end(), arr);
}

// Radix Sort en GPU usando CUDA
#define THREADS_PER_BLOCK 256

__device__ int get_digit(int number, int digit_place) {
    return (number / digit_place) % 10;
}

__global__ void counting_sort_kernel(int* d_input, int* d_output, int* d_count, int n, int digit_place) {
    __shared__ int local_count[10];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < 10) {
        local_count[threadIdx.x] = 0;
    }
    __syncthreads();

    if (thread_id < n) {
        int digit = get_digit(d_input[thread_id], digit_place);
        atomicAdd(&local_count[digit], 1);
    }

    __syncthreads();

    if (threadIdx.x < 10) {
        atomicAdd(&d_count[threadIdx.x], local_count[threadIdx.x]);
    }
}

__global__ void exclusive_scan(int* d_count, int* d_scan) {
    __shared__ int temp[10];
    int thread_id = threadIdx.x;

    if (thread_id < 10) {
        temp[thread_id] = d_count[thread_id];
    }
    __syncthreads();

    for (int stride = 1; stride < 10; stride *= 2) {
        int val = (thread_id >= stride) ? temp[thread_id - stride] : 0;
        __syncthreads();
        temp[thread_id] += val;
        __syncthreads();
    }

    if (thread_id < 10) {
        d_scan[thread_id] = temp[thread_id];
    }
}

__global__ void reorder_kernel(int* d_input, int* d_output, int* d_scan, int n, int digit_place) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id < n) {
        int digit = get_digit(d_input[thread_id], digit_place);
        int position = d_scan[digit]++;
        d_output[position] = d_input[thread_id];
    }
}

void ordenar_radix_sort_gpu_wrapper(vector<int>& host_vector) {
    int n = host_vector.size();
    int* d_input;
    int* d_output;
    int* d_count;
    int* d_scan;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, 10 * sizeof(int));
    cudaMalloc(&d_scan, 10 * sizeof(int));

    cudaMemcpy(d_input, host_vector.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int max_val = *max_element(host_vector.begin(), host_vector.end());

    for (int digit_place = 1; max_val / digit_place > 0; digit_place *= 10) {
        cudaMemset(d_count, 0, 10 * sizeof(int));

        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        counting_sort_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_count, n, digit_place);

        exclusive_scan<<<1, 10>>>(d_count, d_scan);

        reorder_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_scan, n, digit_place);

        swap(d_input, d_output);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_vector.data(), d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaFree(d_scan);
}


void Benchmark(int n, int nt) {

    ofstream outfile("Graficos/output.txt");
    if (!outfile) {
        cerr << "Error al crear el archivo de salida" << endl;
        return;
    }

    int num_iterations = 100; // Número de iteraciones para el benchmark

    for (int iter = 0; iter < num_iterations; iter++) {
        vector<int> vec1(n), vec2(n), vec3(n);
        srand(time(0));
        for (int i = 0; i < n; i++) {
            int val = rand() % 100;
            vec1[i] = val;
            vec2[i] = val;
            vec3[i] = val;
        }

        double start_time, end_time, elapsed_time;

        // Benchmark CPU
        start_time = omp_get_wtime();
        ordenar_radix_sort_cpu_wrapper(vec2.data(), n, nt);
        end_time = omp_get_wtime();
        elapsed_time = end_time - start_time;
        outfile << "CPU Iteración " << iter + 1 << " " << fixed << setprecision(9) << elapsed_time << " segundos" << endl;

        // Benchmark GPU
        start_time = omp_get_wtime();
        ordenar_radix_sort_gpu_wrapper(vec3);
        end_time = omp_get_wtime();
        elapsed_time = end_time - start_time;
        outfile << "GPU Iteración " << iter + 1 << " " << fixed << setprecision(9) << elapsed_time << " segundos" << endl;

        outfile << "----------------------------------------" << endl;
    }

    outfile.close();

}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Uso: " << argv[0] << " <n> <modo>" << endl;
        return 1;
    }

    int n = atoi(argv[1]); // tamaño del arreglo
    int nt = atoi(argv[2]); // cantidad de hilos por defecto
    

    Benchmark(n, nt);
    cout << "Benchmark completado" << endl;
    cout << "Ejecute Python3 Gen_Graficos.py   para que se generen los graficos" << endl;
    
    return 0;
}