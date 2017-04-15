#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>

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

void showFileDetails(InputFileManager& in, BlockManager& bm)
{
	cout << "Dimensions: "
		<< in.getHeight() << " rows X "
		<< in.getWidth() << " cols"
		<< " - Cell size: X=" << in.getCellSizeX()
		<< ", Y=" << in.getCellSizeY()
		<< endl;

	cout << "Projection: " << in.getProjection() << endl;

	cout << "Block size: " << bm.getBlockHeight() << " rows X " << bm.getBlockWidth() << " cols"
		<< " (" << bm.getHeightInBlocks() << " x " << bm.getWidthInBlocks()
		<< " = " << bm.getNumberOfBlocks() << " blocks)"
		<< endl;
}

void runSingleThread(
	CommandLineParser& parser,
	string inputFilePath, string outputFilePath,
	int borderSize, int blockHeight, int blockWidth,
	int& totalTimeReadMs, int& totalTimeCompMs, int& totalTimeWriteMs)
{
	InputFileManager in(inputFilePath);
	OutputFileManager out(outputFilePath,
		in.getGeoTransform(),
		in.getProjection(),
		in.getHeight(), in.getWidth());
	BlockManager bm(in.getHeight(), in.getWidth(), blockHeight, blockWidth);
	GpuBlockProcessor gpu(parser,
		blockHeight + borderSize * 2,
		blockWidth + borderSize * 2,
		in.getCellSizeX(), in.getCellSizeY());

	showFileDetails(in, bm);

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

			totalTimeReadMs += rTime;
			totalTimeCompMs += cTime;
			totalTimeWriteMs += wTime;
		}
		cout << "TR: " << totalTimeReadMs
			<< ", TC: " << totalTimeCompMs
			<< ", TW: " << totalTimeWriteMs
			<< ", T: " << totalTimeReadMs + totalTimeCompMs + totalTimeWriteMs << endl;
	});
}

void runFourThreads(
	CommandLineParser& parser,
	string inputFilePath, string outputFilePath,
	int borderSize, int blockHeight, int blockWidth,
	int& totalTimeReadMs, int& totalTimeCompMs, int& totalTimeWriteMs)
{
	int numberOfThreads = 4;

	InputFileManager in1(inputFilePath);
	InputFileManager in2(inputFilePath);
	InputFileManager in3(inputFilePath);
	InputFileManager in4(inputFilePath);

	OutputFileManager out(outputFilePath,
		in1.getGeoTransform(),
		in1.getProjection(),
		in1.getHeight(), in1.getWidth());
	BlockManager bm(in1.getHeight(), in1.getWidth(), blockHeight, blockWidth);

	GpuBlockProcessor gpu1(parser, blockHeight + borderSize * 2, blockWidth + borderSize * 2, in1.getCellSizeX(), in1.getCellSizeY());
	GpuBlockProcessor gpu2(parser, blockHeight + borderSize * 2, blockWidth + borderSize * 2, in2.getCellSizeX(), in2.getCellSizeY());
	GpuBlockProcessor gpu3(parser, blockHeight + borderSize * 2, blockWidth + borderSize * 2, in3.getCellSizeX(), in3.getCellSizeY());
	GpuBlockProcessor gpu4(parser, blockHeight + borderSize * 2, blockWidth + borderSize * 2, in4.getCellSizeX(), in4.getCellSizeY());

	showFileDetails(in1, bm);

	timeIt("TOTAL: ", [&]() {
		int lastProcessedBlock;
		for (int i = 0; i < bm.getNumberOfBlocks(); i += numberOfThreads)
		{
			if (i + numberOfThreads >= bm.getNumberOfBlocks())
			{
				continue;
			}

			BlockRect rectIn1 = bm.getBlock(i + 0, borderSize, true);
			BlockRect rectIn2 = bm.getBlock(i + 1, borderSize, true);
			BlockRect rectIn3 = bm.getBlock(i + 2, borderSize, true);
			BlockRect rectIn4 = bm.getBlock(i + 3, borderSize, true);

			BlockRect rectOut1 = bm.getBlock(i + 0);
			BlockRect rectOut2 = bm.getBlock(i + 1);
			BlockRect rectOut3 = bm.getBlock(i + 2);
			BlockRect rectOut4 = bm.getBlock(i + 3);

			auto t1 = async([&]() { return timeIt("", [&]() {in1.readBlock(rectIn1, gpu1.getInBlock()); }); });
			auto t2 = async([&]() { return timeIt("", [&]() {in2.readBlock(rectIn2, gpu2.getInBlock()); }); });
			auto t3 = async([&]() { return timeIt("", [&]() {in3.readBlock(rectIn3, gpu3.getInBlock()); }); });
			auto t4 = async([&]() { return timeIt("", [&]() {in4.readBlock(rectIn4, gpu4.getInBlock()); }); });

			int rTime = timeIt("", [&]() {
				int rTime1 = t1.get(); //totalTimeReadMs += rTime1;
				int rTime2 = t2.get(); //totalTimeReadMs += rTime2;
				int rTime3 = t3.get(); //totalTimeReadMs += rTime3;
				int rTime4 = t4.get(); //totalTimeReadMs += rTime4;
			});
			totalTimeReadMs += rTime;

			int cTime1 = timeIt("", [&]() {gpu1.processBlock(rectIn1, rectOut1); }); totalTimeCompMs += cTime1;
			int cTime2 = timeIt("", [&]() {gpu2.processBlock(rectIn2, rectOut2); }); totalTimeCompMs += cTime2;
			int cTime3 = timeIt("", [&]() {gpu3.processBlock(rectIn3, rectOut3); }); totalTimeCompMs += cTime3;
			int cTime4 = timeIt("", [&]() {gpu4.processBlock(rectIn4, rectOut4); }); totalTimeCompMs += cTime4;

			int wTime1 = timeIt("", [&]() {out.writeBlock(rectOut1, gpu1.getOutBlock()); }); totalTimeWriteMs += wTime1;
			int wTime2 = timeIt("", [&]() {out.writeBlock(rectOut2, gpu2.getOutBlock()); }); totalTimeWriteMs += wTime2;
			int wTime3 = timeIt("", [&]() {out.writeBlock(rectOut3, gpu3.getOutBlock()); }); totalTimeWriteMs += wTime3;
			int wTime4 = timeIt("", [&]() {out.writeBlock(rectOut4, gpu4.getOutBlock()); }); totalTimeWriteMs += wTime4;

			cout << setw(3) << fixed << setprecision(0) << i * 100.0 / bm.getNumberOfBlocks() << "% -> "
				<< rectOut1
				<< setw(6) << rTime
				<< setw(6) << (cTime1 + cTime2 + cTime3 + cTime4)
				<< setw(6) << (wTime1 + wTime2 + wTime3 + wTime4) << endl;

			lastProcessedBlock = i + numberOfThreads;
		}

		for (int i = lastProcessedBlock; i < bm.getNumberOfBlocks(); i++)
		{
			BlockRect rectIn1 = bm.getBlock(i + 0, borderSize, true);
			BlockRect rectOut1 = bm.getBlock(i + 0);

			int rTime1 = timeIt("", [&]() {in1.readBlock(rectIn1, gpu1.getInBlock()); }); totalTimeReadMs += rTime1;
			int cTime1 = timeIt("", [&]() {gpu1.processBlock(rectIn1, rectOut1); }); totalTimeCompMs += cTime1;
			int wTime1 = timeIt("", [&]() {out.writeBlock(rectOut1, gpu1.getOutBlock()); }); totalTimeWriteMs += wTime1;

			cout << setw(3) << fixed << setprecision(0) << i * 100.0 / bm.getNumberOfBlocks() << "% -> "
				<< rectOut1
				<< setw(6) << rTime1
				<< setw(6) << cTime1
				<< setw(6) << wTime1 << endl;
		}

		cout << "TR: " << totalTimeReadMs
			<< ", TC: " << totalTimeCompMs
			<< ", TW: " << totalTimeWriteMs
			<< ", T: " << totalTimeReadMs + totalTimeCompMs + totalTimeWriteMs << endl;
	});
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
	int blockHeight = 512, blockWidth = 512;

	if (parser.isIntParameterValid("BlockWidth", 1))
	{
		blockWidth = parser.getIntParameter("BlockWidth");
	}

	if (parser.isIntParameterValid("BlockHeight", 1))
	{
		blockHeight = parser.getIntParameter("BlockHeight");
	}
	std::cout << "Input: " << parser.getInputFileName() << std::endl;
	std::cout << "Output: " << parser.getOutputFileName() << std::endl;

	int totalTimeReadMs = 0;
	int totalTimeCompMs = 0;
	int totalTimeWriteMs = 0;

	try
	{
		GpuBlockProcessor::startCuda();

		if (parser.isIntParameterValid("Threads", 4))
		{
			runFourThreads(parser,
				inputFilePath, outputFilePath,
				borderSize, blockHeight, blockWidth,
				totalTimeReadMs, totalTimeCompMs, totalTimeWriteMs);
		}
		else
		{
			runSingleThread(parser,
				inputFilePath, outputFilePath,
				borderSize, blockHeight, blockWidth,
				totalTimeReadMs, totalTimeCompMs, totalTimeWriteMs);
		}
	}
	catch (runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}

	GpuBlockProcessor::stopCuda();

	return 0;
}