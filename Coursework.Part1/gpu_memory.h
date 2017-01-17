
#pragma once

#include <cuda_runtime.h>

/*
template<typename T>
class DeviceMemory
{
    T* value;
public:
    explicit DeviceMemory(T&& value)
    {
        const auto len = bytes();

        cudaMalloc(static_cast<void**>(&this->value), len);
        cudaMemcpy(this->value, &value, len, cudaMemcpyHostToDevice);
    }

    void copyTo(T* destination)
    {
        cudaMemcpy(destination, value, sizeof(T), cudaMemcpyDeviceToHost);
    }

    operator T*()
    {
        return value;
    }

    constexpr unsigned bytes() { return sizeof(T); }
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
*/