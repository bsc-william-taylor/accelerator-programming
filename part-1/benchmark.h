\
#pragma once

#include <fstream>
#include <ctime>

enum class measure_in
{
    secs = 1000,
    ms = 1
};

template<measure_in measure, unsigned times, typename Functor, typename... Args>
void benchmark(Functor&& method, Args&&... args)
{
    auto typeString = measure == measure_in::ms ? " (ms) " : " (secs) ";
    auto total = 0.0;

    std::ofstream csvFile("benchmark.csv");
    csvFile << "Benchmark Results,  \n";
    csvFile << "ID, Time" << typeString << "\n";
    
    for (auto i = 1u; i <= times; ++i) 
    {
        const auto start = clock();
        method(std::forward<Args>(args)...);
        const auto stop = clock();

        auto time = (stop - start) / double(measure);
        total += time;

        csvFile << i << "," << time << "\n";
    }

    csvFile << "Total" << typeString << ", Average" << typeString << "\n";
    csvFile << total << "," << total / times;
    csvFile.close();
}