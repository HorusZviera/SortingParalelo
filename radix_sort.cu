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

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    #pragma omp parallel for num_threads(nt)
    for (int i = n - 1; i >= 0; i--) {
        int digit = (vec[i] / exp) % 10;
        int pos = count[digit] - 1;
        output[pos] = vec[i];
        count[digit]--;
    }

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
    ofstream outfile_timing("Graficos/tiempo_vs_n.txt");
    ofstream outfile_cpu_speedup("Graficos/aceleracion_cpu_vs_hilos.txt");
    ofstream outfile_cpu_efficiency("Graficos/eficiencia_cpu_vs_hilos.txt");
    ofstream outfile_gpu_speedup("Graficos/aceleracion_gpu_vs_bloques.txt");
    ofstream outfile_gpu_efficiency("Graficos/eficiencia_gpu_vs_bloques.txt");
    ofstream outfile_speedup_vs_n("Graficos/aceleracion_vs_n.txt");

    if (!outfile_timing || !outfile_cpu_speedup || !outfile_cpu_efficiency || !outfile_gpu_speedup || !outfile_gpu_efficiency || !outfile_speedup_vs_n) {
        cerr << "Error al crear archivos de salida" << endl;
        return;
    }

    int max_threads = omp_get_max_threads();
    int max_blocks = 1024; // Asume un límite razonable para bloques CUDA

    for (int i = 1000; i <= n; i *= 10) {
        vector<int> vec_cpu(i), vec_gpu(i);
        srand(time(0));
        for (int j = 0; j < i; j++) {
            int val = rand() % 100;
            vec_cpu[j] = val;
            vec_gpu[j] = val;
        }

        // Tiempo CPU
        double start_time = omp_get_wtime();
        ordenar_radix_sort_cpu_wrapper(vec_cpu.data(), i, nt);
        double cpu_time = omp_get_wtime() - start_time;
        outfile_timing << i << " " << cpu_time << endl;

        // Tiempo GPU
        start_time = omp_get_wtime();
        ordenar_radix_sort_gpu_wrapper(vec_gpu);
        double gpu_time = omp_get_wtime() - start_time;
        outfile_timing << i << " " << gpu_time << endl;

        // Speedup
        outfile_speedup_vs_n << i << " " << cpu_time / gpu_time << endl;
    }

    // CPU Speedup y Eficiencia
    int problem_size = 1000000;
    for (int threads = 1; threads <= max_threads; threads++) {
        vector<int> vec(problem_size);
        srand(time(0));
        for (int i = 0; i < problem_size; i++) {
            vec[i] = rand() % 100;
        }

        double baseline_time;
        if (threads == 1) {
            double start_time = omp_get_wtime();
            ordenar_radix_sort_cpu_wrapper(vec.data(), problem_size, 1);
            baseline_time = omp_get_wtime() - start_time;
        }

        double start_time = omp_get_wtime();
        ordenar_radix_sort_cpu_wrapper(vec.data(), problem_size, threads);
        double elapsed_time = omp_get_wtime() - start_time;

        double speedup = baseline_time / elapsed_time;
        double efficiency = speedup / threads;

        outfile_cpu_speedup << threads << " " << speedup << endl;
        outfile_cpu_efficiency << threads << " " << efficiency << endl;
    }

    // GPU Speedup y Eficiencia
    for (int blocks = 1; blocks <= max_blocks; blocks *= 2) {
        vector<int> vec(problem_size);
        srand(time(0));
        for (int i = 0; i < problem_size; i++) {
            vec[i] = rand() % 100;
        }

        double baseline_time;
        if (blocks == 1) {
            double start_time = omp_get_wtime();
            ordenar_radix_sort_gpu_wrapper(vec);
            baseline_time = omp_get_wtime() - start_time;
        }

        double start_time = omp_get_wtime();
        ordenar_radix_sort_gpu_wrapper(vec);
        double elapsed_time = omp_get_wtime() - start_time;

        double speedup = baseline_time / elapsed_time;
        double efficiency = speedup / blocks;

        outfile_gpu_speedup << blocks << " " << speedup << endl;
        outfile_gpu_efficiency << blocks << " " << efficiency << endl;
    }

    outfile_timing.close();
    outfile_cpu_speedup.close();
    outfile_cpu_efficiency.close();
    outfile_gpu_speedup.close();
    outfile_gpu_efficiency.close();
    outfile_speedup_vs_n.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Uso: " << argv[0] << " <n> <num_threads>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);

    Benchmark(n, nt);
    cout << "Benchmark completado. Resultados guardados en archivos de texto." << endl;
    return 0;
}