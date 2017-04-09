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

#include "BlockRect.h"

using namespace GeoRsGpu;

BlockRect::BlockRect(int rowStart, int colStart, int height, int width)
{
	m_rowStart = rowStart;
	m_colStart = colStart;
	m_height = height;
	m_width = width;
}

bool BlockRect::contains(int rowIndex, int colIndex) const
{
	return rowIndex >= m_rowStart
		&& rowIndex < (m_rowStart + m_height)
		&& colIndex >= m_colStart
		&& colIndex < (m_width + m_colStart);
}

bool BlockRect::operator== (const BlockRect& block) const
{
	return getRowStart() == block.getRowStart()
		&& getColStart() == block.getColStart()
		&& getHeight() == block.getHeight()
		&& getWidth() == block.getWidth();
}
