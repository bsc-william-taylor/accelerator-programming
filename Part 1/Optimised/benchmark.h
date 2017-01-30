
#pragma once

#include <algorithm>
#include <ctime>

enum class measure_in
{
    secs = 1000000,
    ms = 1000
};

template<measure_in measure, typename Functor, typename... Args>
double benchmark(Functor&& method, Args&&... args)
{
    const auto start = clock();
    method(std::forward<Args>(args)...);
    return (clock() - start) / (CLOCKS_PER_SEC / double(measure));
}