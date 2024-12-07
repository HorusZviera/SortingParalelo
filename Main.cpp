#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thread>

using namespace std;

// Global variables
const int tamano = 100;
int* arreglo_A;
int* arreglo_B;

void rellenarArreglos(int tamano) {
    srand(time(0));

    arreglo_A = new int[tamano];
    arreglo_B = new int[tamano];

    // Llenar los arreglos 
    for (int i = 0; i < tamano; ++i) {
        arreglo_A[i] = rand() % 11;
        arreglo_B[i] = arreglo_A[i]; 
    }
}

void imprimirArreglo(int tamano) {
    cout << "Arreglo: ";
    for (int i = 0; i < tamano; ++i) {
        cout << arreglo_A[i] << " ";
    }
    cout << endl;
}

void benchmarkCPU(int n, int nt) {
    // Implementar benchmark en CPU
}

void benchmarkGPU(int n, int nt) {
    // Implementar benchmark en GPU
}

int main(int argc, char* argv[]) {

    int max_threads = thread::hardware_concurrency();


    if (argc != 4) {
        cerr << "Uso: " << argv[0] << " <n> <modo> <nt>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);          // Tamaño de los arreglos
    int modo = atoi(argv[2]);       //0: CPU, 1: GPU
    int nt = atoi(argv[3]);         // Número de threads

    if (n <= 0 || (modo != 0 && modo != 1) || nt <= 0) {
        cerr << "Argumentos invalidos." << endl;
        return 1;
    }

    if (nt > max_threads) {
        cerr << "El número de threads no puede ser mayor al número máximo de threads del sistema (" << max_threads << ")." << endl;
        return 1;
    }


    rellenarArreglos(n);
    cout << "Arreglo generado." << endl;
    imprimirArreglo(n);

    // benchmark
    if (modo == 0) {
        benchmarkCPU(n, nt);
    } else {
        benchmarkGPU(n, nt);
    }
    


    // Liberar memoria
    delete[] arreglo_A;
    delete[] arreglo_B;

    return 0;
}