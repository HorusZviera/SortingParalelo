#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

void ordenar_paralelo(vector<int>& vec) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            sort(vec.begin(), vec.end());
        }
    }
}

int main() {
    const int tamano = 1000000;
    vector<int> vec(tamano);

    // Inicializar el vector con valores aleatorios
    srand(time(0));
    for (int i = 0; i < tamano; ++i) {
        vec[i] = rand();
    }

    // Medir el tiempo tomado por la ordenación paralela
    double tiempo_inicio = omp_get_wtime();
    ordenar_paralelo(vec);
    double tiempo_fin = omp_get_wtime();

    cout << "Tiempo tomado para la ordenación paralela: " << tiempo_fin - tiempo_inicio << " segundos" << endl;

    // Verificar el resultado
    if (is_sorted(vec.begin(), vec.end())) {
        cout << "El vector está ordenado." << endl;
    } else {
        cout << "El vector NO está ordenado." << endl;
    }

    return 0;
}