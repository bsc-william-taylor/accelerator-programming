
#include "gpu_kernels.h"
#include "gpu_memory.h"

#include <iostream>

int main(void)
{  
    HostMemory<NunmericData> a{{10}}, b{{ 10 }}, c{{ 0 }};
    DeviceMemory<NunmericData> da{a}, db(b), dc{c};

    calculate<<<1, 1>>>(da, db, dc);

    dc.copyTo(c);    std::cout << c->number << std::endl;    return std::cin.get();
}