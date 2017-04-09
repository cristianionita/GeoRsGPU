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

#ifndef GEORSGPU_INPUT_FILE_MANAGER_H
#define GEORSGPU_INPUT_FILE_MANAGER_H

#include <string>

#include "gdal_priv.h"

#include "BlockRect.h"

namespace GeoRsGpu {

	class InputFileManager {
	public:
		InputFileManager(std::string filePath);
		~InputFileManager();

		inline int getHeight() const { return m_poRasterBand->GetYSize(); }
		inline int getWidth() const { return m_poRasterBand->GetXSize(); }
		inline double* getGeoTransform() const { return (double*)m_geoTransform; }

		void readBlock(BlockRect rect, BlockRect clippedRect, float* block);

	private:
		GDALDataset* m_poDataset;
		GDALDriver* m_poDriver;
		double m_geoTransform[6];
		GDALRasterBand* m_poRasterBand;
	};

};
#endif // !GEORSGPU_INPUT_FILE_MANAGER_H
