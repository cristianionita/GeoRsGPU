
#include <iostream>

#include "RasterCommand.h"
#include "CommandLineParser.h"
#include "BlockManager.h"
#include "InputFileManager.h"
#include "OutputFileManager.h"
#include "GpuBlockProcessor.cuh"

using namespace GeoRsGpu;
using namespace std;

ostream& operator<< (ostream& out, BlockRect const& block)
{
	out << "[" << block.getRowStart()
		<< "," << block.getColStart()
		<< " " << block.getHeight()
		<< "x" << block.getWidth() << "]";

	return out;
}

#define ELEM(vector, line, column) vector[(line) * nLineSize + (column)]
void showOutBlock(float* out, BlockRect rect)
{
	int nLineSize = rect.getWidth();
	for (int ri = 0; ri < rect.getHeight(); ri++)
	{
		for (int ci = 0; ci < rect.getWidth(); ci++)
		{
			std::cout << ELEM(out, ri, ci) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

#include <chrono>
void timeIt(std::string operationName, std::function<void()> func)
{
	auto startTime = std::chrono::high_resolution_clock::now();
	func();
	auto endTime = std::chrono::high_resolution_clock::now();


	int totalMilliseconds = (int)std::chrono::
		duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

	std::cout << "[" << operationName << "]: " << totalMilliseconds << "ms" << std::endl;
}

int main(int argc, char *argv[])
{
	/*CommandLineParser parser(argc, argv);

	if (!parser.isValid())
	{
		return 1;
	}

	std::cout << "Command: "
		<< static_cast<std::underlying_type<RasterCommand>::type>(parser.getCommand())
		<< std::endl;
	std::cout << "Input: " << parser.getInputFileName() << std::endl;
	std::cout << "Output: " << parser.getOutputFileName() << std::endl;

	if (parser.isIntParameterValid("BlockWidth", 0))
	{
		std::cout << "BlockWidth: " << parser.getIntParameter("BlockWidth") << std::endl;
	}
	else
	{
		std::cout << "BlockWidth not found - using default value" << std::endl;
	}*/

	string inputFilePath = "d:\\GeoGPUTeste\\Data\\dem9112.tif";
	//string inputFilePath = "d:\\Dropbox\\ASE\\GIS\\EUD_CP-DEMS_4500025000-AA.tif";
	//string inputFilePath = "d:\\temp\\small_1_100_dem.tif";
	string outputFilePath = "d:\\temp\\georsgpu_result.tif";
	RasterCommand command = RasterCommand::Aspect;

	int borderSize = 1;
	int blockHeight = 3000, blockWidth = 2500;

	try
	{
		InputFileManager in(inputFilePath);		
		//OutputFileManager out(outputFilePath, in.getGeoTransform(), 10, 10);		
		//BlockRect rectOut(0, 0, 10, 10);
		//float block[] = {
		//	0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		//	10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
		//	20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
		//	30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
		//	40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
		//	50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
		//	60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
		//	70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
		//	80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
		//	90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
		//};
		//out.writeBlock(rectOut, block);

		OutputFileManager out(outputFilePath, 
			in.getGeoTransform(), in.getHeight(), in.getWidth());
		BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);
		GpuBlockProcessor gpu(command, 
			blockHeight + borderSize * 2,
			blockWidth + borderSize * 2);

		for (int i = 0; i < bm.getNumberOfBlocks(); i++)
		{
			BlockRect rectIn = bm.getBlock(i, borderSize, true);
			BlockRect rectOut = bm.getBlock(i);

			//cout << rectOut << endl;
			//timeIt("R", [&]() {in.readBlock(rectIn, gpu.getInBlock()); });
			//timeIt("C", [&]() {gpu.processBlock(rectIn, rectOut); });
			//timeIt("W", [&]() {out.writeBlock(rectOut, gpu.getOutBlock()); });

			in.readBlock(rectIn, gpu.getInBlock());
			timeIt("C", [&]() {gpu.processBlock(rectIn, rectOut); });
			out.writeBlock(rectOut, gpu.getOutBlock());
		}
	}
	catch (runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}

	return 0;
}