
#pragma once

#include <cuda_runtime.h>

template<typename T>
class DeviceMemory
{
    T* value;
public:
    explicit DeviceMemory(T&& value)
    {
        cudaMalloc(static_cast<void**>(&this->value), sizeof(T));
        cudaMemcpy(this->value, &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyTo(T* destination)
    {
        cudaMemcpy(destination, value, sizeof(T), cudaMemcpyDeviceToHost);
    }

    operator T*()
    {
        return value;
    }
};

template<typename T>
class HostMemory
{
    T value;
public:
    explicit HostMemory(T&& value) : 
        value(value)
    {
    }

    operator T*()
    {
        return &value;
    }

    operator T()
    {
        return value;
    }

    T* operator ->() {
        return static_cast<T*>(&value);
    }
};