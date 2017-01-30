
#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

namespace cuda
{   
    using cudaErrorHandler = std::function<void(int, std::string)>;
    using cudaDeviceList = std::vector<cudaDeviceProp>;
    using cudaDeviceFilter = std::function<cudaDeviceProp(cudaDeviceList&)>;

    cudaDeviceProp findDevice(cudaDeviceFilter functor = [](cudaDeviceList& list){ return list.front(); });
    cudaError errorCheck(cudaErrorHandler functor = [](int, std::string){});
    
    void debugInfo(const cudaDeviceProp& prop);
   
    template<int B, int T, typename K, typename... Args>
    void start(K kernel, Args&&... args)
    {
        kernel<<<B, T>>>(std::forward<Args>(args)...);
    }

    template<typename T>
    class memory 
    {
        T pointer;
        size_t sz;
    public:
        memory(T src, size_t bytes);
        memory(const memory& copy) = delete;
        memory(memory&& copy) = delete;

        virtual ~memory();

        operator T();

        size_t size();

        void transfer(T dest);
        void assign(T src);
        void allocate();
        void release();
    };

    template<typename T>
    memory<T>::memory(T source, size_t bytes)
        : pointer(nullptr), sz(bytes)
    {
        allocate();
        assign(source);
    }

    template<typename T>
    memory<T>::~memory()
    {
        release();
    }

    template <typename T>
    void memory<T>::allocate()
    {
        if(pointer == nullptr)
        {
            cudaMalloc(&pointer, size());
        }
    }

    template <typename T>
    void memory<T>::release()
    {
        if (pointer != nullptr)
        {
            cudaFree(&pointer);
            pointer = nullptr;
        }
    }

    template<typename T>
    memory<T>::operator T()
    {
        return pointer;
    }

    template<typename T>
    void memory<T>::transfer(T destination)
    {
        if (destination != nullptr)
        {
            cudaMemcpy(destination, pointer, sz, cudaMemcpyDeviceToHost);
        }
    }

    template <typename T>
    void memory<T>::assign(T source)
    {
        if (source != nullptr)
        {
            cudaMemcpy(pointer, source, sz, cudaMemcpyHostToDevice);
        }
    }

    template<typename T>
    size_t memory<T>::size()
    {
        return sz;
    }
}