#pragma once

#include <fstream>
#include <chrono>

template<int C>
class csv
{
    std::ofstream output;
public:
    explicit csv(const std::string& name, const std::string& title) :
        output(name.c_str())
    {
        output << title.c_str();
        for (auto i = 1u; i < C; ++i)
            output << ",";
        output << std::endl;
    }

    template<typename V, typename... Args>
    void append(V&& first, Args&&... args)
    {
        append(first);
        append(args...);
    }

    template<typename V>
    void append(V&& v)
    {
        output << v;
    }

    void append_row(const std::string& row)
    {
        output << row.c_str() << std::endl;
    }
};

template<unsigned times, typename Functor, typename... Args>
void benchmark(const char* fn, Functor&& method, Args&&... args)
{
    using namespace std::chrono;
    csv<2> table(fn, "Benchmark Results");
    table.append_row("ID, Time (ms)");
    
    auto total = 0.0, millseconds = 0.0;
 
    for (auto i = 1u; i <= times; ++i) 
    {
        const auto start = high_resolution_clock::now();
        method(std::forward<Args>(args)...);
        const auto stop = high_resolution_clock::now();
        millseconds = duration_cast<milliseconds>(stop - start).count();

        table.append(i, ",", millseconds, "\n");
        total += millseconds;
    }

    table.append_row("Total (ms), Average (ms) \n");
    table.append(total, ",", total / times);
}