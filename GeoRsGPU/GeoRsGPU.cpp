
#include <iostream>

#include "RasterCommand.h"
#include "CommandLineParser.h"
#include "BlockManager.h"
#include "InputFileManager.h"
#include "OutputFileManager.h"
#include "GpuBlockProcessor.cuh"

using namespace GeoRsGpu;
using namespace std;

ostream& operator<< (ostream& out, BlockRect& const block)
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
	string outputFilePath = "d:\\temp\\georsgpu_result.tif";
	RasterCommand command = RasterCommand::Slope;

	int borderSize = 1;
	int blockHeight = 800, blockWidth = 700;
	try
	{
		InputFileManager in(inputFilePath);		
		OutputFileManager out(outputFilePath, 
			in.getGeoTransform(), in.getHeight(), in.getWidth());
		BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);
		GpuBlockProcessor gpu(command, 
			blockHeight + borderSize * 2,
			blockWidth + borderSize * 2);

		for (int i = 0; i < bm.getNumberOfBlocks(); i++)
		{
			BlockRect rect = bm.getBlock(i, borderSize, false);
			BlockRect rectClipped = bm.getBlock(i, borderSize, true);
			//cout << rectClipped << " -> " << bm.getBlock(i) << endl;
			in.readBlock(rect, rectClipped, gpu.getInBlock());
			gpu.processBlock(rectClipped);
						
			out.writeBlock(bm.getBlock(i), gpu.getOutBlock());
		}
	}
	catch (runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}

	return 0;
}