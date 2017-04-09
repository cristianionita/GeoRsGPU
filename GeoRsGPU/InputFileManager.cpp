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

#include "gdal_priv.h"

#include "InputFileManager.h"

using namespace GeoRsGpu;

InputFileManager::InputFileManager(std::string filePath)
{
	GDALAllRegister();

	m_poDataset = (GDALDataset*)GDALOpen(filePath.c_str(), GA_ReadOnly);
	if (m_poDataset == NULL)
	{
		throw std::runtime_error("Failed to open GDAL dataset.");
	}

	m_poDriver = GetGDALDriverManager()->GetDriverByName(m_poDataset->GetDriverName());
	if (m_poDriver == NULL)
	{
		throw std::runtime_error("Failed to identify the GDAL driver.");
	}

	// In the particular, but common, case of a "north up" image without any rotation or shearing, 
	// the georeferencing transform takes the following form :
	// 0 -> top left x
	// 1 -> w-e pixel resolution
	// 2 -> 0
	// 3 -> top left y
	// 4 -> 0
	// 5 -> n-s pixel resolution (negative value)	
	m_poDataset->GetGeoTransform(m_geoTransform);

	// We assume that the file has only one raster band
	// (we always use the first one).
	m_poRasterBand = m_poDataset->GetRasterBand(1);
}

InputFileManager::~InputFileManager()
{
	GDALClose(m_poDataset);
}

void InputFileManager::readBlock(const BlockRect rect, float * const __restrict block)
{
	// Read data from file
	CPLErr result = m_poRasterBand->RasterIO(
		GDALRWFlag::GF_Read,
		rect.getColStart(), rect.getRowStart(),
		rect.getWidth(), rect.getHeight(),
		block,
		rect.getWidth(), rect.getHeight(),
		GDALDataType::GDT_Float32, 0, 0);

	if (result != CE_None)
	{
		throw std::runtime_error("Error reading from the input file.");
	}
}