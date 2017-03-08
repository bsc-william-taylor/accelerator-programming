#include "CppUnitTest.h"
#include "../Cpu/ppm.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{		
	TEST_CLASS(OutputTests)
	{
	public:
        double compare(std::vector<std::uint8_t>& img1, std::vector<std::uint8_t>& img2)
        {
            auto len = img1.size();
            auto total = 0;

            for (auto i = 0; i < len; ++i)
            {
                int pixelMaxDifference = std::numeric_limits<std::uint8_t>::max();
                total += ((double)abs(img1[i] - img2[i]) / (double)pixelMaxDifference) * 100.0;
            }

            return (double)total / (double)len;
        }

		TEST_METHOD(MatchingOutput)
		{
            ppm cpuImage("../Cpu/out.ppm"), gpuImage("../Gpu/cl-out.ppm");
            double percentage = compare(cpuImage.data, gpuImage.data);

            std::wstringstream ss;
            ss << "Different Percentage %";
            ss << percentage;
            ss << std::endl;

            Assert::IsTrue(percentage < 1.0f, ss.str().c_str());
		}
	};
}