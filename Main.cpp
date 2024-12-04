#include <iostream>

// Function declarations for CPU and GPU sorting
void sortCPU(int* data, int size);
void sortGPU(int* data, int size);

using namespace std;

int main() {
    const int tamano = 10;
    int datos[tamano] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    cout << "Datos originales: ";
    for (int i = 0; i < tamano; ++i) {
        cout << datos[i] << " ";
    }
    cout << endl;

    // Ordenar usando CPU
    sortCPU(datos, tamano);
    cout << "Datos ordenados (CPU): ";
    for (int i = 0; i < tamano; ++i) {
        cout << datos[i] << " ";
    }
    cout << endl;

    // Restablecer datos
    int datos2[tamano] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    // Ordenar usando GPU
    sortGPU(datos2, tamano);
    cout << "Datos ordenados (GPU): ";
    for (int i = 0; i < tamano; ++i) {
        cout << datos2[i] << " ";
    }
    cout << endl;

    return 0;
}