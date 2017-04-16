#include <iostream>
#include <iomanip>
#include <chrono>

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
	CommandLineParser parser(argc, argv);

	if (!parser.isValid())
	{
		return 1;
	}

	string inputFilePath = parser.getInputFileName();
	string outputFilePath = parser.getOutputFileName();
	RasterCommand command = parser.getCommand();

	int borderSize = 1;
	int blockHeight = 4096, blockWidth = 4096;

	if (parser.isIntParameterValid("BlockWidth", 1))
	{
		blockWidth = parser.getIntParameter("BlockWidth");
	}

	if (parser.isIntParameterValid("BlockHeight", 1))
	{
		blockHeight = parser.getIntParameter("BlockHeight");
	}

	try
	{
		InputFileManager in(inputFilePath);
		OutputFileManager out(outputFilePath,
			in.getGeoTransform(),
			in.getProjection(),
			in.getHeight(), in.getWidth());
		BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);

		std::cout << "Input: " << parser.getInputFileName() << std::endl;
		std::cout << "Output: " << parser.getOutputFileName() << std::endl;

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

		GpuBlockProcessor gpu(parser,
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