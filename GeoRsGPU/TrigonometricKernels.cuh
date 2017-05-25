
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

#ifndef TRIGONOMETRIC_KERNELS_H
#define TRIGONOMETRIC_KERNELS_H

#include "cuda_runtime.h"

namespace GeoRsGpu {

	//__device__ const float RadDegree = 57.29578f;

	struct KernelATAN
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return atan(e);
		}
	};


	struct KernelATANH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return atanh(e);
		}
	};


	struct KernelACos
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return acos(e);
		}
	};


	struct KernelACosH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return acosh(e);
		}
	};



	struct KernelASin
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return asin(e);
		}
	};


	struct KernelASinH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return asinh(e);
		}
	};


	/*struct KernelATAN2
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return atan(e);
		}
	};*/


	struct KernelTAN
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return tan(e);
		}
	};


	struct KernelTANH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return tanh(e);
		}
	};


	struct KernelCos
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return cos(e);
		}
	};


	struct KernelCosH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return cosh(e);
		}
	};


	struct KernelSin
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return sin(e);
		}
	};


	struct KernelSinH
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			return sinh(e);
		}
	};

}
#endif // !TRIGONOMETRIC_KERNELS_H