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

extern "C" void ordenar_paralelo_wrapper(int* arr, int size) {
    vector<int> vec(arr, arr + size);
    ordenar_paralelo(vec);
    copy(vec.begin(), vec.end(), arr);
}