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

#include "OutputFileManager.h"

using namespace GeoRsGpu;

OutputFileManager::OutputFileManager(std::string filePath,
	double* geoTransform, 
	const char* geoProjection,
	int height, int width)
{
	// Delete output file if exists
	struct stat fileStat;
	if (stat(filePath.c_str(), &fileStat) == 0)
	{
		if (remove(filePath.c_str()) != 0)
		{
			throw std::runtime_error("Error deleting existing output file.");
		}
	}

	// Open file using GDAL
	GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	m_poDataset = poDriver->Create(
		filePath.c_str(), width, height, 1, GDT_Float32, NULL);
	m_poDataset->SetGeoTransform(geoTransform);

	m_poDataset->SetProjection(geoProjection);

	m_poRasterBand = m_poDataset->GetRasterBand(1);
}

OutputFileManager::~OutputFileManager()
{
	GDALClose(m_poDataset);
}

void OutputFileManager::writeBlock(BlockRect rect, float* block)
{
	CPLErr result = m_poRasterBand->RasterIO(
		GDALRWFlag::GF_Write,
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