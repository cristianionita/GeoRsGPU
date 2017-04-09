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

#define ELEM(vector, line, column) vector[(line) * width + (column)]

__global__ void slopeZevenbergen(
	const float * const __restrict input, float * const __restrict output,
	const int height, const int width,
	const int heightOut, const int widthOut,
	const int deltaRow, const int deltaCol,
	const float cellSizeX, const float cellSizeY, const float radDegree)
{
	int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	int colIndexOut = colIndex + deltaCol;
	int rowIndexOut = rowIndex + deltaRow;	

	if (colIndexOut < 0 || colIndexOut >= widthOut
		|| rowIndexOut < 0 || rowIndexOut >= heightOut)
	{
		// We don't have any output value
		return;
	}

	float& outputElem = output[rowIndexOut * widthOut + colIndexOut];

	if (colIndex > 0 && colIndex < width - 1
		&& rowIndex > 0 && rowIndex < height - 1)
	{
		// We have everything we need
		float a = ELEM(input, rowIndex - 1, colIndex - 1);
		float b = ELEM(input, rowIndex - 1, colIndex);
		float c = ELEM(input, rowIndex - 1, colIndex + 1);

		float d = ELEM(input, rowIndex, colIndex - 1);
		float e = ELEM(input, rowIndex, colIndex);
		float f = ELEM(input, rowIndex, colIndex + 1);

		float g = ELEM(input, rowIndex + 1, colIndex - 1);
		float h = ELEM(input, rowIndex + 1, colIndex);
		float i = ELEM(input, rowIndex + 1, colIndex + 1);

		float dZdY = (b - h) / (2 * cellSizeY);
		float dZdX = (f - d) / (2 * cellSizeX);
		outputElem = (sqrt(pow(dZdX, 2) + pow(dZdY, 2))) * radDegree;
	}
	else if (
		colIndex == 0 || rowIndex == 0 || 
		colIndex == width - 1 || rowIndex == height - 1)
	{
		// We are on the edge - we don't have all surrounding values
		outputElem = 0;
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

void GpuBlockProcessor::processBlock(BlockRect rectIn, BlockRect rectOut)
{
	size_t blockSizeBytes = rectIn.getWidth() * rectIn.getHeight() * sizeof(float);
	executeCuda([&]() { return cudaMemcpy(
		m_devIn, m_in, blockSizeBytes, cudaMemcpyHostToDevice); });
	executeCuda([&]() { return cudaDeviceSynchronize(); });

	dim3 grid;
	dim3 block(16, 16);
	grid.x = rectIn.getWidth() / block.x + (rectIn.getWidth() % block.x == 0 ? 0 : 1);
	grid.y = rectIn.getHeight() / block.y + (rectIn.getHeight() % block.y == 0 ? 0 : 1);
	
	const float cellSizeX = 25.0f;
	const float cellSizeY = 25.0f;
	const float radDegree = 57.29578f;

	slopeZevenbergen <<<grid, block>>> (
		m_devIn, m_devOut, 
		rectIn.getHeight(), rectIn.getWidth(),
		rectOut.getHeight(), rectOut.getWidth(),
		rectIn.getRowStart() - rectOut.getRowStart(),
		rectIn.getColStart() - rectOut.getColStart(),
		cellSizeX, cellSizeY, radDegree);

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