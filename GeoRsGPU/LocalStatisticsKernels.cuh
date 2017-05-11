// MIT License
// 
// Copyright(c) 2017 Cristian Ionita, Ionut Sandric, Marian Dardala, Titus Felix Furtuna
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LOCAL_STATISTICS_KERNELS_H
#define LOCAL_STATISTICS_KERNELS_H

#include "cuda_runtime.h"

namespace GeoRsGpu {

	//__device__ const float RadDegree = 57.29578f;

	struct KernelMean
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float sumWindow = 0;
			for (int m = 0; m < 9; m++)
			{
				sumWindow += window[m];
			}
			return sumWindow / 9.0f;
		}
	};

	struct KernelStandardDeviation
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float meanWindow = (a + b + c + d + e + f + g + h + i) / 9.0f;
			float stdevWindow = sqrt((pow(a - meanWindow, 2) + pow(b - meanWindow, 2) + pow(c - meanWindow, 2) + pow(d - meanWindow, 2) + pow(e - meanWindow, 2) + pow(f - meanWindow, 2) + pow(g - meanWindow, 2) + pow(h - meanWindow, 2) + pow(i - meanWindow, 2)) / 9.0f);
			return stdevWindow;
		}
	};

	struct KernelRange
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float min = window[0];
			float max = window[0];
			for (int m = 0; m < 9; m++)
			{
				min = min > int(window[m]) ? window[m] : min;
				max = max < int(window[m]) ? window[m] : max;
			}
			return max - min;
		}
	};
	//MAXIMUM
	struct KernelMaximum
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g, h, i };
			float max = window[0];
			for (int m = 0; m < 9; m++)
			{
				max = max < int(window[m]) ? window[m] : max;
			}
			return max;
		}
	};
	//MEDIAN
	struct KernelMedian
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g, h, i };
			for (int m = 0; m < 9; m++)
			{
				int k = m;
				for (int j = m - 1; k > 0; k--, j--)
				{
					if (window[j] > window[k])
					{
						float temp = window[j];
						window[j] = window[k];
						window[k] = temp;
					}
				}
			}
			return window[4];
		}
	};
	//MINIMUM
	struct KernelMinimum
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float min = window[0];
			for (int m = 0; m < 9; m++)
			{
				min = min > int(window[m]) ? window[m] : min;
			}
			return min;
		}
	};
	//MAJORITY
	struct KernelMajority
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h), int(i) };
			int maxIndex = 0;
			float majValue = 0;
			float maxValue = 0;
			int uniqueValues[9];
			int countValues[9];
			int k = 0;

			for (int m = 0; m < 9; m++)
			{
				int found = 0;
				for (int n = 0; n < m; n++) {
					if (window[m] == window[n])
					{
						found++;
					}
				}
				if (found == 0)
				{
					int count = 1;
					for (int n = m + 1; n < 9; n++) {
						if (window[m] == window[n])
						{
							count++;
						}
					}
					uniqueValues[k] = window[m];
					countValues[k] = count;
					k++;
				}
			}
			int majIndex = countValues[0];

			for (int p = 0; p < k; p++)
			{
				majIndex = majIndex < int(countValues[p]) ? countValues[p] : majIndex;
			}

			for (int p = 0; p < k; p++)
			{

				if (countValues[p] == majIndex)
				{
					maxIndex++;
				}
			}

			if (maxIndex > 1)
			{
				return -9999;
			}
			else
			{
				for (int p = 0; p < k; p++)
				{
					if (countValues[p] == majIndex)
					{
						majValue = uniqueValues[p];
					}
				}
				return majValue;
			}
		}
	};

	//MINORITY
	struct  KernelMinority
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h), int(i) };
			int minIndex = 0;
			float mijValue = 0;
			float minValue = 0;
			int uniqueValues[9];
			int countValues[9];
			int k = 0;

			for (int m = 0; m < 9; m++)
			{
				int found = 0;
				for (int n = 0; n < m; n++) {
					if (window[m] == window[n])
					{
						found++;
					}
				}
				if (found == 0)
				{
					int count = 1;
					for (int n = m + 1; n < 9; n++) {
						if (window[m] == window[n])
						{
							count++;
						}
					}
					uniqueValues[k] = window[m];
					countValues[k] = count;
					k++;
				}
			}
			int mijIndex = countValues[0];

			for (int p = 0; p < k; p++)
			{
				mijIndex = mijIndex > int(countValues[p]) ? countValues[p] : mijIndex;
			}

			for (int p = 0; p < k; p++)
			{

				if (countValues[p] == mijIndex)
				{
					minIndex++;
				}
			}

			if (minIndex > 1)
			{
				return -9999;
			}
			else
			{
				for (int p = 0; p < k; p++)
				{
					if (countValues[p] == mijIndex)
					{
						mijValue = uniqueValues[p];
					}
				}
				return mijValue;
			}
		}
	};

	//SUM
	struct KernelSum
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float sum = 0;
			for (int m = 0; m < 9; m++)
			{
				sum += window[m];
			}
			return sum;
		}
	};

	//VARIETY
	struct KernelVariety
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			int counter = 0;
			for (int m = 0; m < 9; m++)
			{
				int found = 0;
				for (int n = 0; n < m; n++)
				{
					if (int(window[m]) == int(window[n]))
					{
						found++;
					}
				}
				if (found == 0)
				{
					counter++;
				}
			}
			return counter;
		}
	};

	//PERCENTILE
	struct KernelPercentile
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			int count = 0;
			for (int m = 0; m < 9; m++)
			{
				if (window[m] < e)
				{
					count++;
				}
			}
			return count * 100.0f / 8.0f;
		}
	};

	//DIFFERENCE FROM MEAN
	struct KernelDiffFromMean
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			//float window[9] = { a, b, c, d, e, f, g , h, i };
			float meanWindow = (a + b + c + d + e + f + g + h + i) / 9.0f;
			return (e - meanWindow);
		}
	};

	//DEVIATION FROM MEAN
	struct KernelDevFromMean
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float meanWindow = (a + b + c + d + e + f + g + h + i) / 9.0f;
			float stdevWindow = sqrt((pow(a - meanWindow, 2) + pow(b - meanWindow, 2) + pow(c - meanWindow, 2) + pow(d - meanWindow, 2) + pow(e - meanWindow, 2) + pow(f - meanWindow, 2) + pow(g - meanWindow, 2) + pow(h - meanWindow, 2) + pow(i - meanWindow, 2)) / 9.0f);
			return (e - meanWindow) / stdevWindow;
		}
	};
}
#endif // !LOCAL_STATISTICS_KERNELS_H