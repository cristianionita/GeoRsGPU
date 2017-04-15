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

#ifndef GEORSGPU_GPU_BLOCK_PROCESSOR_H
#define GEORSGPU_GPU_BLOCK_PROCESSOR_H

#include <functional>

#include "cuda_runtime.h"

#include "BlockRect.h"
#include "RasterCommand.h"

namespace GeoRsGpu {

	class GpuBlockProcessor {
	public:
		
		GpuBlockProcessor(RasterCommand command,
			int maxBlockHeight, int maxBlockWidth,
			float cellSizeX, float cellSizeY);
		~GpuBlockProcessor();

		inline float* getInBlock() { return m_in; }
		inline float* getOutBlock() { return m_out; }

		void processBlock(BlockRect rectIn, BlockRect rectOut);

	private:

		RasterCommand m_command;
		int m_maxBlockHeight;
		int m_maxBlockWidth;

		float m_cellSizeX;
		float m_cellSizeY;

		float* m_in;
		float* m_out;
		float* m_devIn;
		float* m_devOut;

		void executeCuda(std::function<cudaError_t()> cudaFunc);
		void checkCuda();
	};

};
#endif // !GEORSGPU_GPU_BLOCK_PROCESSOR_H

