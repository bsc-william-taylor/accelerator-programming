
#pragma once

#include <string>

namespace fs
{
    template<int C>
    class csv
    {
        std::ofstream out;
    public:
        csv(const std::string& name, const std::string& title)
        {
            out.open(name);
            out << title;  
  
            for (auto i = 1u; i < C; ++i)
            {
                out << ",";
            }
                
            out << std::endl;
        }

        ~csv()
        {
            out.close();
        }

        template<typename V>
        void append(V&& v)
        {
            out << v;
        }

        template<typename V, typename... Args>
        void append(V&& first, Args&&... args)
        {
            append(first);
            append(args...);
        }

        void append_row(const std::string& row)
        {
            out << row << std::endl;
        }
    };
}
