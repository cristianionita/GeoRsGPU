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

#ifndef GEORSGPU_BLOCK_MANAGER_H
#define GEORSGPU_BLOCK_MANAGER_H

#include "BlockRect.h"

namespace GeoRsGpu {

	class BlockManager {
	public:
		BlockManager(int height, int width, int blockHeight, int blockWidth);

		inline int getHeight() const { return m_height; }
		inline int getWidth() const { return m_width; }
		
		inline int getBlockHeight() const { return m_blockHeight; }
		inline int getBlockWidth() const { return m_blockWidth; }
		
		inline int getHeightInBlocks() const
		{
			int heightInBlocks = m_height / m_blockHeight;
			if (m_height % m_blockHeight != 0)
			{
				heightInBlocks = heightInBlocks + 1;
			}
			return heightInBlocks;
		}
		inline int getWidthInBlocks() const
		{
			int widthInBlocks = m_width / m_blockWidth;
			if (m_width % m_blockWidth != 0)
			{
				widthInBlocks = widthInBlocks + 1;
			}
			return widthInBlocks;
		}
		inline int getNumberOfBlocks() const
		{
			return getHeightInBlocks() * getWidthInBlocks();
		}

		inline int getBlockRowIndex(int blockIndex) const
		{
			return blockIndex / getWidthInBlocks();
		}
		inline int getBlockColIndex(int blockIndex) const
		{
			return blockIndex % getWidthInBlocks();
		}

		inline BlockRect getBlock(int blockIndex, int borderSize = 0, bool clipToFile = false)
		{
			return getBlock(
				getBlockRowIndex(blockIndex),
				getBlockColIndex(blockIndex),
				borderSize,
				clipToFile);
		}
		BlockRect getBlock(int blockRowIndex, int blockColIndex, 
			int borderSize = 0, bool clipToFile = false);

		inline bool isEdgeBlock(int blockIndex) {
			return isEdgeBlock(
				getBlockRowIndex(blockIndex),
				getBlockColIndex(blockIndex));
		}

		inline bool isEdgeBlock(int blockRowIndex, int blockColIndex) {
			return blockRowIndex == 0 || blockRowIndex == getHeightInBlocks() - 1
				|| blockColIndex == 0 || blockColIndex == getWidthInBlocks() - 1;
		}

	private:
		int m_height, m_width, m_blockHeight, m_blockWidth;
	};

};
#endif // !GEORSGPU_BLOCK_MANAGER_H

