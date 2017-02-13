
#include "CppUnitTest.h"
#include "ppm.h"

ppm::file cudaOutput("../Cuda/gpu-mandelbrot.ppm");
ppm::file cpuOutput("../Cpu/cpu-mandelbrot.ppm");

namespace Tests
{
    using Microsoft::VisualStudio::CppUnitTestFramework::Assert;

    TEST_CLASS(FileOutputTests)
    {
    public:
        TEST_METHOD(SameWidth)
        {
            Assert::IsTrue(cudaOutput.is_open() && cpuOutput.is_open(), L"Error Loading Files!");
            auto equalWidth = cudaOutput.compare(ppm::section::Width, cpuOutput);
            Assert::IsTrue(equalWidth, L"Error Images do not have the same width!");
        }

        TEST_METHOD(SameHeight)
        {
            Assert::IsTrue(cudaOutput.is_open() && cpuOutput.is_open(), L"Error Loading Files!");
            auto equalHeight = cudaOutput.compare(ppm::section::Height, cpuOutput);
            Assert::IsTrue(equalHeight, L"Error Images do not have the same height!");
        }

        TEST_METHOD(SameMagicNumber)
        {
            Assert::IsTrue(cudaOutput.is_open() && cpuOutput.is_open(), L"Error Loading Files!");
            auto equalNum = cudaOutput.compare(ppm::section::MagicNum, cpuOutput);
            Assert::IsTrue(equalNum, L"Error Images do not have the same magic number!");
        }

        TEST_METHOD(SameData)
        {
            Assert::IsTrue(cudaOutput.is_open() && cpuOutput.is_open(), L"Error Loading Files!");
            auto equalData = cudaOutput.compare(ppm::section::RawData, cpuOutput);
            Assert::IsTrue(equalData, L"Error Images do not have the same data!");
        }
    };
}