
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

#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include "cuda_runtime.h"

namespace GeoRsGpu {

	//__device__ const float RadDegree = 57.29578f;

	struct KernelAbs
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return fabs(e);
		}
	};

	struct KernelFloor
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return floorf(e);
		}
	};

	struct KernelCeil
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return ceil(e);
		}
	};

	struct KernelExpf
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return expf(e);
		}
	};

	struct KernelExp10f
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return powf(2, 10);
		}
	};

	struct KernelExp2f
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return exp2f(e);
		}
	};

	struct KernelLogf
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return logf(e);
		}
	};

	struct KernelLog10f
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return log10f(e);
		}
	};

	struct KernelLog2f
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return log2f(e);
		}
	};

	struct KernelNegate
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return e * -1;
		}
	};

	struct KernelPower
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return pow(e, a1);
		}
	};

	struct KernelSquare
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return pow(e, 2);
		}
	};

	struct KernelSquareRoot
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY, const float a1, const float a2, const float a3, const float a4, const float a5)
		{
			return sqrt(e);
		}
	};
}

#endif // !MATH_KERNELS_H