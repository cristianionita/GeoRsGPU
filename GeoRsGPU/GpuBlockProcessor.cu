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

#include <stdexcept>

#include "GpuBlockProcessor.cuh"

using namespace GeoRsGpu;

#define ELEM(vector, line, column) vector[(line) * nLineSize + (column)]

__global__ void slopeZevenbergen(
	const float * const __restrict input, float * const __restrict output,
	const int nNumberOfLines, const int nLineSize,
	const float cellSizeX, const float cellSizeY, const float radDegree)
{
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int lineIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if (colIndex > 0 && colIndex < nLineSize - 1
		&& lineIndex > 0 && lineIndex < nNumberOfLines - 1)
	{
		//float a = ELEM(input, lineIndex - 1, colIndex - 1);
		//float b = ELEM(input, lineIndex - 1, colIndex);
		//float c = ELEM(input, lineIndex - 1, colIndex + 1);

		//float d = ELEM(input, lineIndex, colIndex - 1);
		//float e = ELEM(input, lineIndex, colIndex);
		//float f = ELEM(input, lineIndex, colIndex + 1);

		//float g = ELEM(input, lineIndex + 1, colIndex - 1);
		//float h = ELEM(input, lineIndex + 1, colIndex);
		//float i = ELEM(input, lineIndex + 1, colIndex + 1);

		//float dZdY = (b - h) / (2 * cellSizeY);
		//float dZdX = (f - d) / (2 * cellSizeX);
		//ELEM(output, lineIndex, colIndex) = (sqrt(pow(dZdX, 2) + pow(dZdY, 2))) * radDegree;
		ELEM(output, lineIndex, colIndex) = 1;
	}
	else if (
		colIndex == 0 || lineIndex == 0 || 
		colIndex == nLineSize - 1 || lineIndex == nNumberOfLines - 1)
	{
		ELEM(output, lineIndex, colIndex) = 2;
	}
}

const int MAX_ERROR_MESSAGE_LEN = 300;

GpuBlockProcessor::GpuBlockProcessor(RasterCommand command,
	int maxBlockHeight, int maxBlockWidth)
{
	m_command = command;
	m_maxBlockHeight = maxBlockHeight;
	m_maxBlockWidth = maxBlockWidth;

	/*** Init CUDA ***/
	// Choose which GPU to run on, change this on a multi-GPU system.
	executeCuda([]() { return cudaSetDevice(0); });
	checkCuda();

	executeCuda([]() { return cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); });
	checkCuda();

	/*** Allocate host and device memory for two blocks (in / out) ***/
	executeCuda([&]() { return cudaMallocHost((void**)&m_in,
		maxBlockHeight * maxBlockWidth * sizeof(float)); });
	checkCuda();

	executeCuda([&]() { return cudaMallocHost((void**)&m_out,
		maxBlockHeight * maxBlockWidth * sizeof(float)); });
	checkCuda();

	executeCuda([&]() { return cudaMalloc((void**)&m_devIn,
		maxBlockHeight * maxBlockWidth * sizeof(float)); });
	checkCuda();

	executeCuda([&]() { return cudaMalloc((void**)&m_devOut,
		maxBlockHeight * maxBlockWidth * sizeof(float)); });
	checkCuda();
}

GpuBlockProcessor::~GpuBlockProcessor()
{
	executeCuda([&]() { return cudaFreeHost((void**)m_in); });
	executeCuda([&]() { return cudaFreeHost((void**)m_out); });
	executeCuda([&]() { return cudaFree((void**)m_devIn); });
	executeCuda([&]() { return cudaFree((void**)m_devOut); });
	executeCuda([&]() { return cudaDeviceSynchronize(); });

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	executeCuda([]() { return cudaDeviceReset(); });
}

void GpuBlockProcessor::processBlock(BlockRect rect)
{
	size_t blockSizeBytes = rect.getWidth() * rect.getHeight() * sizeof(float);
	executeCuda([&]() { return cudaMemcpy(
		m_devIn, m_in, blockSizeBytes, cudaMemcpyHostToDevice); });
	executeCuda([&]() { return cudaDeviceSynchronize(); });

	dim3 grid;
	dim3 block(16, 16);
	grid.x = rect.getWidth() / block.x + (rect.getWidth() % block.x == 0 ? 0 : 1);
	grid.y = rect.getHeight() / block.y + (rect.getHeight() % block.y == 0 ? 0 : 1);
	
	const float cellSizeX = 25.0f;
	const float cellSizeY = 25.0f;
	const float radDegree = 57.29578f;

	slopeZevenbergen <<<grid, block>>> (
		m_devIn, m_devOut, rect.getHeight(), rect.getWidth(), cellSizeX, cellSizeY, radDegree);
	executeCuda([&]() { return cudaDeviceSynchronize(); });

	executeCuda([&]() { return cudaMemcpy(
		m_out, m_devOut, blockSizeBytes, cudaMemcpyDeviceToHost); });
	executeCuda([&]() { return cudaDeviceSynchronize(); });
}

void GpuBlockProcessor::executeCuda(std::function<cudaError_t()> cudaFunc)
{
	cudaError_t code = cudaFunc();
	if (code != cudaSuccess)
	{
		char buffer[MAX_ERROR_MESSAGE_LEN];
		snprintf(buffer, sizeof(buffer),
			"CUDA #%d - %s", code, cudaGetErrorString(code));
		throw std::runtime_error(buffer);
	}
}

void GpuBlockProcessor::checkCuda()
{
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		char buffer[MAX_ERROR_MESSAGE_LEN];
		snprintf(buffer, sizeof(buffer),
			"CUDA #%d - %s", code, cudaGetErrorString(code));
		throw std::runtime_error(buffer);
	}
}