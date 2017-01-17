
#include "gpu_kernels.h"
#include "gpu_memory.h"

#include <iostream>
#include <locale>

void cudaCheckError()
{
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cin.get();
}

int main(void)
{  
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    //std::cout << props.name << std::endl;
    //std::cout << props.clockRate << std::endl;
    //std::cin.get();
    
    int numbers1[2024];
    int numbers2[2024];
    int numbers3[2024];

    int* deviceNumbers1, *deviceNumbers2, *deviceNumbers3;

    memset(numbers1, 10, sizeof(numbers1));
    memset(numbers2, 50, sizeof(numbers2));
    memset(numbers3, 0, sizeof(numbers3));

    cudaMalloc((void**)&deviceNumbers1, sizeof(numbers1));
    cudaMalloc((void**)&deviceNumbers2, sizeof(numbers1));
    cudaMalloc((void**)&deviceNumbers3, sizeof(numbers1));

    cudaMemcpy(deviceNumbers1, numbers1, sizeof(numbers1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNumbers2, numbers2, sizeof(numbers2), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNumbers3, numbers3, sizeof(numbers3), cudaMemcpyHostToDevice);
    
    //cudaCheckError();
    
    vector_add<<<2024, 1>>>(deviceNumbers1, deviceNumbers2, deviceNumbers3);
    cudaMemcpy(numbers3, deviceNumbers3, sizeof(numbers3), cudaMemcpyDeviceToHost);    std::cout << numbers3[0] << std::endl;    std::cin.get();    cudaFree(deviceNumbers1);    cudaFree(deviceNumbers2);    cudaFree(deviceNumbers3);    return 0;
}