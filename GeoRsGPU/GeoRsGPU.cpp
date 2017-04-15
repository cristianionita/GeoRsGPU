#include <iostream>
#include <iomanip>

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
	out << "[" << setw(6) << block.getRowStart()
		<< "," << setw(6) << block.getColStart()
		<< " " << setw(6) << block.getHeight()
		<< " x" << setw(6) << block.getWidth() << "]";

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
int timeIt(std::string operationName, std::function<void()> func)
{
	auto startTime = std::chrono::high_resolution_clock::now();
	func();
	auto endTime = std::chrono::high_resolution_clock::now();


	int totalMilliseconds = (int)std::chrono::
		duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

	if (operationName.length() > 0)
	{
		std::cout << "[" << operationName << "]: " << totalMilliseconds << "ms" << std::endl;
	}

	return totalMilliseconds;
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
	RasterCommand command = RasterCommand::Hillshade;

	int borderSize = 1;
	int blockHeight = 3000, blockWidth = 2500;

	try
	{
		InputFileManager in(inputFilePath);
		OutputFileManager out(outputFilePath,
			in.getGeoTransform(),
			in.getProjection(),
			in.getHeight(), in.getWidth());
		BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);

		cout << "Raster: " << inputFilePath << endl;
		cout << "Dimensions: "
			<< in.getHeight() << " rows X "
			<< in.getWidth() << " cols"
			<< " - Cell size: X=" << in.getCellSizeX()
			<< ", Y=" << in.getCellSizeY()
			<< endl;
		cout << "Block size: " << blockHeight << " rows X " << blockWidth << " cols"
			<< " (" << bm.getHeightInBlocks() << " x " << bm.getWidthInBlocks()
			<< " = " << bm.getNumberOfBlocks() << " blocks)"
			<< endl;

		GpuBlockProcessor gpu(command,
			blockHeight + borderSize * 2,
			blockWidth + borderSize * 2,
			in.getCellSizeX(), in.getCellSizeY());

		timeIt("TOTAL: ", [&]() {
			for (int i = 0; i < bm.getNumberOfBlocks(); i++)
			{
				BlockRect rectIn = bm.getBlock(i, borderSize, true);
				BlockRect rectOut = bm.getBlock(i);
				
				int rTime = timeIt("", [&]() {in.readBlock(rectIn, gpu.getInBlock()); });
				int cTime = timeIt("", [&]() {gpu.processBlock(rectIn, rectOut); });
				int wTime = timeIt("", [&]() {out.writeBlock(rectOut, gpu.getOutBlock()); });

				cout << setw(3) << fixed << setprecision(0) << i * 100.0 / bm.getNumberOfBlocks() << "% -> "
					<< rectOut 
					<< setw(6) << rTime 
					<< setw(6) << cTime 
					<< setw(6) << wTime << endl;
			}
		});
	}
	catch (runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}

	return 0;
}