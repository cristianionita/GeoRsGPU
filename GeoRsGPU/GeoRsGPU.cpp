
#include <iostream>

#include "RasterCommand.h"
#include "CommandLineParser.h"
#include "BlockManager.h"
#include "InputFileManager.h"
#include "OutputFileManager.h"

using namespace GeoRsGpu;
using namespace std;

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

	int borderSize = 1;
	int blockHeight = 200, blockWidth = 400;
	try
	{
		InputFileManager in(inputFilePath);		
		OutputFileManager out(outputFilePath, 
			in.getGeoTransform(), in.getHeight(), in.getWidth());
		BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);

		float* block = new float[(blockHeight + borderSize * 2) * (blockWidth + borderSize * 2)];

		for (int i = 0; i < bm.getNumberOfBlocks(); i++)
		{
			BlockRect rect = bm.getBlock(i, borderSize, false);
			BlockRect rectClipped = bm.getBlock(i, borderSize, true);
			in.readBlock(rect, rectClipped, block);
			out.writeBlock(rectClipped, block);
			/*if (bm.isEdgeBlock(i)) {
				cout << "Processing " << i
					<< "(" << bm.getBlockRowIndex(i) << "," << bm.getBlockColIndex(i) << ")"
					<< " ("
					<< bm.getBlock(i, 2, false).getRowStart()
					<< ","
					<< bm.getBlock(i, 2, false).getColStart()
					<< ","
					<< bm.getBlock(i, 2, false).getHeight()
					<< ","
					<< bm.getBlock(i, 2, false).getWidth()
					<< ")"
					<< " ("
					<< bm.getBlock(i, 2, true).getRowStart()
					<< ","
					<< bm.getBlock(i, 2, true).getColStart()
					<< ","
					<< bm.getBlock(i, 2, true).getHeight()
					<< ","
					<< bm.getBlock(i, 2, true).getWidth()
					<< ")"
					<< endl;
			}*/
		}

		delete[] block;
	}
	catch (runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}

	return 0;
}