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
#include "DemKernels.cuh"
#include "LocalStatisticsKernels.cuh"
#include "TrigonometricKernels.cuh"
#include "MathKernels.cuh"

using namespace GeoRsGpu;

const int MAX_ERROR_MESSAGE_LEN = 300;

GpuBlockProcessor::GpuBlockProcessor(
	CommandLineParser& parser,
	int maxBlockHeight, int maxBlockWidth,
	float cellSizeX, float cellSizeY)
	: m_commandLineParser(parser)
{
	m_command = m_commandLineParser.getCommand();
	m_maxBlockHeight = maxBlockHeight;
	m_maxBlockWidth = maxBlockWidth;

	m_cellSizeX = cellSizeX;
	m_cellSizeY = cellSizeY;

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
}

template <typename EQ>
__global__ void gpuKernel(
	const float * const __restrict input, float * const __restrict output,
	const int height, const int width,
	const int heightOut, const int widthOut,
	const int deltaRow, const int deltaCol,
	const float cellSizeX, const float cellSizeY)
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
		float a = input[width * (rowIndex - 1) + colIndex - 1];
		float b = input[width * (rowIndex - 1) + colIndex];
		float c = input[width * (rowIndex - 1) + colIndex + 1];

		float d = input[width * rowIndex + colIndex - 1];
		float e = input[width * rowIndex + colIndex];
		float f = input[width * rowIndex + colIndex + 1];

		float g = input[width * (rowIndex + 1) + colIndex - 1];
		float h = input[width * (rowIndex + 1) + colIndex];
		float i = input[width * (rowIndex + 1) + colIndex + 1];

		outputElem = EQ()(a, b, c, d, e, f, g, h, i, cellSizeX, cellSizeY);
	}
	else if (
		colIndex == 0 || rowIndex == 0 ||
		colIndex == width - 1 || rowIndex == height - 1)
	{
		// We are on the edge - we don't have all surrounding values
		outputElem = 0;
	}
}
template <typename EQ>
__global__ void gpuKernel_NXM(
	const float * const __restrict input, float * const __restrict output,
	const int height, const int width,
	const int heightOut, const int widthOut,
	const int deltaRow, const int deltaCol,
	const float cellSizeX, const float cellSizeY)
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
		float a = input[width * (rowIndex - 1) + colIndex - 1];
		float b = input[width * (rowIndex - 1) + colIndex];
		float c = input[width * (rowIndex - 1) + colIndex + 1];

		float d = input[width * rowIndex + colIndex - 1];
		float e = input[width * rowIndex + colIndex];
		float f = input[width * rowIndex + colIndex + 1];

		float g = input[width * (rowIndex + 1) + colIndex - 1];
		float h = input[width * (rowIndex + 1) + colIndex];
		float i = input[width * (rowIndex + 1) + colIndex + 1];

		outputElem = EQ()(a, b, c, d, e, f, g, h, i, cellSizeX, cellSizeY);
	}
	else if (
		colIndex == 0 || rowIndex == 0 ||
		colIndex == width - 1 || rowIndex == height - 1)
	{
		// We are on the edge - we don't have all surrounding values
		outputElem = 0;
	}
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

	// input and output sizes (different for edge cases)
	int inH = rectIn.getHeight(), inW = rectIn.getWidth();
	int outH = rectOut.getHeight(), outW = rectOut.getWidth();

	// delta row and column - specify how the output block
	// is positioned against the input block (non zero for edge cases)
	int dR = rectIn.getRowStart() - rectOut.getRowStart();
	int dC = rectIn.getColStart() - rectOut.getColStart();

#define KERNEL_PARAMS m_devIn, m_devOut, inH, inW, outH, outW, dR, dC, m_cellSizeX, m_cellSizeY

	switch (m_command)
	{
	case RasterCommand::Slope:
	{
		bool useBurruogh = true; // use Burruogh by default
		if (m_commandLineParser.parameterExists("Alg"))
		{
			if (m_commandLineParser.getStringParameter("Alg") == "Zvn")
			{
				useBurruogh = false;
			}
			else if (m_commandLineParser.getStringParameter("Alg") == "Brr")
			{
				useBurruogh = true;
			}
			else
			{
				throw std::runtime_error("Invalid slope algorithm.");
			}
		}
		if (useBurruogh)
		{
			gpuKernel<KernelSlopeBurruogh> << <grid, block >> > (KERNEL_PARAMS);
		}
		else
		{
			gpuKernel<KernelSlopeZevenbergen> << <grid, block >> > (KERNEL_PARAMS);
		}
	}
	break;

	case RasterCommand::Hillshade:
		gpuKernel<KernelHillshade> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Aspect:
		gpuKernel<KernelAspect> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::TotalCurvature:
		gpuKernel<KernelTotalCurvature> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::PlanCurvature:
		gpuKernel<KernelPlanCurvature> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::ProfileCurvature:
		gpuKernel<KernelProfileCurvature> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::TopographicPositionIndex:
	{
		bool useEP = true;
		bool useDIF = false;
		bool useDEV = false;
		if (m_commandLineParser.parameterExists("Alg"))
		{
			if (m_commandLineParser.getStringParameter("Alg") == "EP")
			{
				useEP = true;
			}
			else if (m_commandLineParser.getStringParameter("Alg") == "DIF")
			{
				useDIF = true;
			}
			else if (m_commandLineParser.getStringParameter("Alg") == "DEV")
			{
				useDEV = true;
			}
			else
			{
				throw std::runtime_error("Invalid topographic postion index");
			}
		}
		if (useEP)
		{
			gpuKernel<KernelTopographicPositionIndex_EP> << <grid, block >> > (KERNEL_PARAMS);
		}
		else if (useDIF)
		{
			gpuKernel<KernelTopographicPositionIndex_DIF> << <grid, block >> > (KERNEL_PARAMS);
		}
		else if (useDEV)
		{
			gpuKernel<KernelTopographicPositionIndex_DEV> << <grid, block >> > (KERNEL_PARAMS);
		}
	}
	break;


	case RasterCommand::Minority:
		gpuKernel<KernelMinority> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Majority:
		gpuKernel<KernelMajority> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Mean:
		gpuKernel<KernelMean> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Median:
		gpuKernel<KernelMedian> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Maximum:
		gpuKernel<KernelMaximum> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Minimum:
		gpuKernel<KernelMinimum> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::StandardDeviation:
		gpuKernel<KernelStandardDeviation> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Range:
		gpuKernel<KernelRange> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Variety:
		gpuKernel<KernelVariety> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Percentile:
		gpuKernel<KernelPercentile> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::DiffFromMean:
		gpuKernel<KernelDiffFromMean> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::StDevFromMean:
		gpuKernel<KernelDevFromMean> << <grid, block >> > (KERNEL_PARAMS);
		break;


		//Trigonometric kernels
	case RasterCommand::ATan:
		gpuKernel<KernelATAN> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::ATanH:
		gpuKernel<KernelATANH> << <grid, block >> > (KERNEL_PARAMS);
		break;

		//case RasterCommand::ATan2:
		//	gpuKernel<KernelATAN2> << <grid, block >> > (KERNEL_PARAMS);
		//	break;

	case RasterCommand::ACos:
		gpuKernel<KernelACos> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::ACosH:
		gpuKernel<KernelACosH> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::ASin:
		gpuKernel<KernelASin> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::ASinH:
		gpuKernel<KernelASinH> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Tan:
		gpuKernel<KernelTAN> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::TanH:
		gpuKernel<KernelTANH> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Cos:
		gpuKernel<KernelCos> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::CosH:
		gpuKernel<KernelCosH> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Sin:
		gpuKernel<KernelSin> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::SinH:
		gpuKernel<KernelSinH> << <grid, block >> > (KERNEL_PARAMS);
		break;


		//Math functions
	case RasterCommand::Abs:
		gpuKernel<KernelAbs> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::RoundUp:
		gpuKernel<KernelCeil> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::RoundDown:
		gpuKernel<KernelFloor> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Exp10f:
		gpuKernel<KernelExp10f> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Exp2f:
		gpuKernel<KernelExp2f> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Lnf:
		gpuKernel<KernelLogf> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Log10f:
		gpuKernel<KernelLog10f> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Log2f:
		gpuKernel<KernelLog2f> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Negate:
		gpuKernel<KernelNegate> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Power:
		gpuKernel<KernelPower> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::Square:
		gpuKernel<KernelSquare> << <grid, block >> > (KERNEL_PARAMS);
		break;

	case RasterCommand::SqRoot:
		gpuKernel<KernelSquareRoot> << <grid, block >> > (KERNEL_PARAMS);
		break;



	default:
		char buffer[MAX_ERROR_MESSAGE_LEN];
		snprintf(buffer, sizeof(buffer),
			"Command #%d is not supported.", m_command);
		throw std::runtime_error(buffer);
	}

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

void GpuBlockProcessor::startCuda()
{
	/*** Init CUDA ***/
	// Choose which GPU to run on, change this on a multi-GPU system.
	executeCuda([]() { return cudaSetDevice(0); });
	checkCuda();

	executeCuda([]() { return cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); });
	checkCuda();
}

void GpuBlockProcessor::stopCuda()
{
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	executeCuda([]() { return cudaDeviceReset(); });
}
