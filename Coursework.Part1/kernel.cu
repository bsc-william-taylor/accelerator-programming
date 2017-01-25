
#include <iostream>

#include "cuda-utilities.h"
#include "cuda-kernels.h"

const auto onError = [](auto code, auto msg) 
{
    std::cout << msg << std::endl;
    std::cin.get();
};

int main()
{    
   int numbers1[2024]{10};
   int numbers2[2024]{50};
   int numbers3[2024]{0};

   const auto bytes = sizeof(numbers1);
   const auto device = cuda::findDevice();

   cuda::debugInfo(device);
   cuda::memory<int*> array1{numbers1, bytes};
   cuda::memory<int*> array2{numbers2, bytes};
   cuda::memory<int*> array3{numbers3, bytes};
   cuda::errorCheck(onError);

   cuda::start<1, 1>(add, array1, array2, array3);   cuda::errorCheck(onError);

   array3.transfer(numbers3);
   std::cout << numbers3[0] << std::endl;   std::cin.get();   return 0;
}