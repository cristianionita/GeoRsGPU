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

#include <algorithm>

#include "BlockManager.h"

using namespace GeoRsGpu;

BlockManager::BlockManager(int height, int width, int blockHeight, int blockWidth)
{
	m_height = height;
	m_width = width;
	
	m_blockHeight = blockHeight;
	m_blockWidth = blockWidth;	
}

BlockRect BlockManager::getBlock(int blockRowIndex, int blockColIndex,
	int borderSize, bool clipToFile)
{
	//Debug.Assert(borderSize <= BlockLineSize);
	//Debug.Assert(borderSize <= BlockNumberOfLines);

	// Get position and dimensions without border
	int startRow = blockRowIndex * getBlockHeight();
	int startCol = blockColIndex * getBlockWidth();
	int blockHeight = std::min(getBlockHeight(), getHeight() - startRow);
	int blockWidth = std::min(getBlockWidth(), getWidth() - startCol);

	// Expand to include border
	startRow = startRow - borderSize;
	startCol = startCol - borderSize;
	blockHeight = blockHeight + 2 * borderSize;
	blockWidth = blockWidth + 2 * borderSize;

	// Shrink for edge blocks
	if (clipToFile)
	{
		if (blockRowIndex == 0)
		{
			// we don't have upper border
			startRow = startRow + borderSize;
			blockHeight = blockHeight - borderSize;
		}
		if (blockRowIndex == getHeightInBlocks() - 1)
		{
			// we don't have bottom border
			blockHeight = blockHeight - borderSize;
		}

		if (blockColIndex == 0)
		{
			// we don't have left border
			startCol = startCol + borderSize;
			blockWidth = blockWidth - borderSize;
		}
		if (blockColIndex == getWidthInBlocks() - 1)
		{
			// we don't have right border
			blockWidth = blockWidth - borderSize;
		}
	}

	return BlockRect(startRow, startCol, blockHeight, blockWidth);
}
