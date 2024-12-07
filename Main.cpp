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
    int count[10] = {0};

    #pragma omp parallel for num_threads(nt)
    for (int i = 0; i < n; i++)
        count[(vec[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    #pragma omp parallel for num_threads(nt)
    for (int i = n - 1; i >= 0; i--) {
        output[count[(vec[i] / exp) % 10] - 1] = vec[i];
        count[(vec[i] / exp) % 10]--;
    }

    #pragma omp parallel for num_threads(nt)
    for (int i = 0; i < n; i++)
        vec[i] = output[i];
}

void radix_sort_cpu(vector<int>& vec, int nt) {
    int max_val = *max_element(vec.begin(), vec.end());

    for (int exp = 1; max_val / exp > 0; exp *= 10)
        counting_sort_cpu(vec, exp, nt);
}

extern "C" void ordenar_radix_sort_cpu_wrapper(int* arr, int size, int nt) {
    vector<int> vec(arr, arr + size);
    double start_time = omp_get_wtime();
    radix_sort_cpu(vec, nt);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    copy(vec.begin(), vec.end(), arr);
    cout << "Tiempo de ejecución en CPU: " << elapsed_time << " segundos" << endl;
    cout << "Arreglo ordenado en CPU: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Radix Sort en GPU usando CUDA
__global__ void counting_sort_kernel(int* d_vec, int* d_output, int* d_count, int n, int exp) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(&d_count[(d_vec[idx] / exp) % 10], 1);
    }
    __syncthreads();

    if (idx < 10) {
        for (int i = 1; i < 10; i++) {
            atomicAdd(&d_count[i], d_count[i - 1]);
        }
    }
    __syncthreads();

    if (idx < n) {
        int pos = atomicSub(&d_count[(d_vec[idx] / exp) % 10], 1) - 1;
        d_output[pos] = d_vec[idx];
    }
}

void radix_sort_gpu(vector<int>& vec) {
    int n = vec.size();
    int* d_vec;
    int* d_output;
    int* d_count;
    cudaMalloc(&d_vec, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, 10 * sizeof(int));

    cudaMemcpy(d_vec, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int max_val = *max_element(vec.begin(), vec.end());
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        cudaMemset(d_count, 0, 10 * sizeof(int));
        counting_sort_kernel<<<(n + 255) / 256, 256>>>(d_vec, d_output, d_count, n, exp);
        cudaMemcpy(d_vec, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(vec.data(), d_vec, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);
    cudaFree(d_output);
    cudaFree(d_count);
}

extern "C" void ordenar_radix_sort_gpu_wrapper(int* arr, int size) {
    vector<int> vec(arr, arr + size);
    double start_time = omp_get_wtime();
    radix_sort_gpu(vec);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    copy(vec.begin(), vec.end(), arr);
    cout << "Tiempo de ejecución en GPU: " << elapsed_time << " segundos" << endl;
    cout << "Arreglo ordenado en GPU: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}




int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Uso: " << argv[0] << " <n> <modo> <nt>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int modo = atoi(argv[2]);
    int nt = atoi(argv[3]);


    // Generar arreglo aleatorio
    vector<int> vec(n);
    srand(time(0));
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 100;
    }

    // Se despliega menu con las opciones
    while (true) {
        cout << "Ordenamiento con algoritmo Radix:" << endl;
        cout << "0 - Ordenamiento por CPU" << endl;
        cout << "1 - Ordenamiento por GPU" << endl;
        cout << "2 - Imprimir arreglo" << endl;
        cout << "3 - Salir" << endl;
        cout << "Ingrese su opción: ";
        cin >> modo;
        switch (modo) {
            case 0:
                cout << "Modo CPU seleccionado." << endl;
                ordenar_radix_sort_cpu_wrapper(vec.data(), n, nt);
                break;
            case 1:
                cout << "Modo GPU seleccionado." << endl;
                ordenar_radix_sort_gpu_wrapper(vec.data(), n);
                break;
            case 2:
                cout << "Arreglo inicial: ";
                for (int i = 0; i < n; i++) {
                    cout << vec[i] << " ";
                }
                cout << endl;
                break;
            case 3:
                cout << "Saliendo..." << endl;
                return 0;
            default:
                cerr << "Modo no válido. Use 0 para CPU, 1 para GPU, 2 para imprimir el arreglo, o 3 para salir." << endl;
        }
    }
    return 0;
}