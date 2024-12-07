#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

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

        #pragma omp for
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
        #pragma omp critical
        {
            output[count[digit] - 1] = vec[i];
            count[digit]--;
        }
    }

    // Paso 4: Copiar al arreglo original
    #pragma omp parallel for num_threads(nt)
    for (int i = 0; i < n; i++) {
        vec[i] = output[i];
    }
}

void radix_sort_cpu(vector<int>& vec, int nt) {
    int max_val = *max_element(vec.begin(), vec.end());

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_cpu(vec, exp, nt);
    }
}

extern "C" void ordenar_radix_sort_cpu_wrapper(int* arr, int size, int nt) {
    vector<int> vec(arr, arr + size);
    double start_time = omp_get_wtime();
    radix_sort_cpu(vec, nt);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    copy(vec.begin(), vec.end(), arr);
    cout << "Tiempo de ejecuci\u00f3n en CPU: " << elapsed_time << " segundos" << endl;
    cout << "Arreglo ordenado en CPU: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Radix Sort en GPU usando CUDA

#define THREADS_PER_BLOCK 256

__device__ int get_digit(int number, int digit_place) {
    return (number / digit_place) % 10;
}

__global__ void counting_sort_kernel(int* d_input, int* d_output, int* d_count, int n, int digit_place) {
    __shared__ int local_count[10];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_offset = threadIdx.x;

    if (thread_offset < 10) {
        local_count[thread_offset] = 0;
    }
    __syncthreads();

    if (thread_id < n) {
        int digit = get_digit(d_input[thread_id], digit_place);
        atomicAdd(&local_count[digit], 1);
    }
    __syncthreads();

    if (thread_offset < 10) {
        atomicAdd(&d_count[thread_offset], local_count[thread_offset]);
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
        int val = 0;
        if (thread_id >= stride) {
            val = temp[thread_id - stride];
        }
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
        int position = d_scan[digit] + atomicAdd(&d_scan[digit], 1);  // Incrementar en el mismo tiempo
        d_output[position] = d_input[thread_id];
    }
}


void ordenar_radix_sort_gpu_wrapper(std::vector<int>& host_vector) {
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

        std::swap(d_input, d_output);
    }

    cudaMemcpy(host_vector.data(), d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaFree(d_scan);
}




int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Uso: " << argv[0] << " <n> <modo> <nt>" << endl;
        return 1;
    }

    // validar numero de hilos
    int max_threads = omp_get_max_threads();
    if (atoi(argv[3]) > max_threads) {
        cerr << "Error: La cantidad de núcleos no puede sobrepasar " << max_threads << " en este computador." << endl;
        return 1;
    }

    int n = atoi(argv[1]); // tamaño del arreglo
    int modo = atoi(argv[2]); // modo de ejecución
    int nt = atoi(argv[3]); // cantidad de hilos


    vector<int> vec1(n), vec2(n), vec3(n);
    srand(time(0));
    for (int i = 0; i < n; i++) {
        int val = rand() % 100;
        vec1[i] = val;
        vec2[i] = val;
        vec3[i] = val;
    }

    cout << "Arreglo original: ";
    for (int i = 0; i < n; i++) {
        cout << vec1[i] << " ";
    }
    cout << endl;

    cout << "----------------------------------------" << endl;
    cout << vec2.data() << endl;
    cout << "Modo CPU." << endl;
    ordenar_radix_sort_cpu_wrapper(vec2.data(), n, nt);

    cout << "----------------------------------------" << endl;

    cout << "Modo GPU." << endl;
    ordenar_radix_sort_gpu_wrapper(vec3);
    cout << "Arreglo ordenado en GPU: ";
    for (int i = 0; i < n; i++) {
        cout << vec3[i] << " ";
    }
    cout << endl;

    cout << "----------------------------------------" << endl;

    return 0;
}
