
#pragma once

#include <algorithm>
#include <fstream>
#include <ctime>

enum class measure_in
{
    secs = 1000000,
    ms = 1000
};

template<measure_in measure, unsigned times, typename Functor, typename... Args>
void benchmark(Functor&& method, Args&&... args)
{
    auto type = measure == measure_in::ms ? " (ms) " : " (secs) ";
    auto total = 0.0;

    std::ofstream csv("benchmark.csv");
    csv << "Benchmark Results \n";
    csv << "ID, Time" << type << ", \n";
    
    for (auto i = 1u; i <= times; ++i) 
    {
        const auto start = clock();
        method(std::forward<Args>(args)...);
        const auto stop = clock();

        auto time = (stop - start) / (CLOCKS_PER_SEC / double(measure));
        csv << i << "," << time << "\n";
        total += time;
    }

    csv << "Total," << type << "Average" << type << "\n";
    csv << total << "," << total / times;
    csv.close();
}