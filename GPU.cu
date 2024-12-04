#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void testKernel(int *d_data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_data[idx] = idx;
}

void verificarErrorCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cerr << "Error de CUDA: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int tamanoArray = 256;
    const int tamanoBloque = 16;
    const int numBloques = (tamanoArray + tamanoBloque - 1) / tamanoBloque;

    int h_data[tamanoArray];
    int *d_data;

    // Asignar memoria en el dispositivo
    verificarErrorCuda(cudaMalloc((void**)&d_data, tamanoArray * sizeof(int)), "Fallo al asignar memoria en el dispositivo");

    // Lanzar kernel
    testKernel<<<numBloques, tamanoBloque>>>(d_data);
    verificarErrorCuda(cudaGetLastError(), "Fallo en el lanzamiento del kernel");

    // Copiar datos de vuelta al host
    verificarErrorCuda(cudaMemcpy(h_data, d_data, tamanoArray * sizeof(int), cudaMemcpyDeviceToHost), "Fallo al copiar datos del dispositivo al host");

    // Verificar resultados
    for (int i = 0; i < tamanoArray; ++i) {
        if (h_data[i] != i) {
            cerr << "Fallo en la verificación en el índice " << i << ": " << h_data[i] << " != " << i << endl;
            return EXIT_FAILURE;
        }
    }

    cout << "¡Prueba pasada!" << endl;

    // Liberar memoria del dispositivo
    verificarErrorCuda(cudaFree(d_data), "Fallo al liberar memoria del dispositivo");

    return EXIT_SUCCESS;
}