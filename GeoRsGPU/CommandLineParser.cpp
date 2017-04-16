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

#include <iostream>
#include <limits>
#include <vector>

#include <args.hxx>

#include "CommandLineParser.h"

using namespace GeoRsGpu;

CommandLineParser::CommandLineParser(int argc, char *argv[])
{
	std::cerr << "GeoRsGPU - v1.0.100" << std::endl << std::endl;

	std::unordered_map<std::string, RasterCommand> map{
		{ "slope", RasterCommand::Slope},
		{ "hillshade", RasterCommand::Hillshade },
		{ "aspect", RasterCommand::Aspect },
		{ "totalCurvature", RasterCommand::TotalCurvature },
		{ "planCurvature", RasterCommand::PlanCurvature },
		{ "profileCurvature", RasterCommand::ProfileCurvature },
		{ "tpi", RasterCommand::TopographicPositionIndex }
	};

	
	args::ArgumentParser parser(getDescription());
	parser.Prog("GeoRsGPU.exe");

	args::Group group(parser, "Required arguments:", args::Group::Validators::All);

	args::MapPositional<std::string, RasterCommand> posCommand(group,
		"command",
		"Raster command to be executed. One of: slope, hillshade",
		map);

	args::Positional<std::string> posInputFile(group,
		"input", "The path to the input file.");
	args::Positional<std::string> posOutputFile(group,
		"output", "The path to the output file.");

	args::PositionalList<std::string> parameters(parser,
		"parameters", "Optional parameters list.");

	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

	try
	{
		parser.ParseCLI(argc, argv);
		
		m_inputFileName = posInputFile.Get();
		m_outputFileName = posOutputFile.Get();

		m_command = posCommand.Get();

		for (auto paramStr : parameters.Get())
		{
			auto eqIndex = paramStr.find_first_of("=");
			if (eqIndex == std::string::npos)
			{
				std::cerr << "ERROR parsing parameter '" << paramStr << "'"
					<< std::endl
					<< "See 'GeoRsGPU --help' for documentation"
					<< std::endl;
				m_isValid = false;
				return;
			}

			std::string paramName = paramStr.substr(0, eqIndex);
			std::string paramValue = paramStr.substr(
				eqIndex + 1, paramStr.length() - eqIndex - 1);

			m_parameters[trim(paramName)] = trim(paramValue);
		}
	}
	catch (args::Help)
	{
		std::cout << parser;
		m_isValid = false;
		return;
	}
	catch (args::ParseError e)
	{
		std::cerr << "ERROR parsing arguments: " << e.what()
			<< std::endl
			<< "See 'GeoRsGPU --help' for documentation"
			<< std::endl;

		m_isValid = false;
		return;
	}
	catch (args::ValidationError e)
	{
		std::cerr << "ERROR parsing arguments: " << e.what()
			<< std::endl << std::endl
			<< "See 'GeoRsGPU --help' for documentation"
			<< std::endl;

		m_isValid = false;
		return;
	}

	m_isValid = true;
}

bool CommandLineParser::isIntParameterValid(std::string paramName, int minValue)
{
	if (!parameterExists(paramName))
	{
		return false;
	}

	try {
		int value = std::stoi(getStringParameter(paramName));
		if (value < minValue)
		{
			return false;
		}

		return true;
	}
	catch (std::invalid_argument&) {
		// if no conversion could be performed
		return false;
	}
	catch (std::out_of_range&) {
		// if the converted value would fall out of the range of the result type 
		// or if the underlying function (std::strtol or std::strtoull) sets errno 
		// to ERANGE.
		return false;
	}
	catch (...) {
		// everything else
		return false;
	}
}

std::string CommandLineParser::getDescription()
{
	return
		"GeoRsGPU - GPU accelerated raster processing (requires NVIDIA CUDA)",
		"\n"
		"Command-specific documentation\n"
		"\n"
		"\n"
		"slope:\n"
		"   Documentation for slope. Parameter: 'Alg' with possible values 'Brr' - Burruogh (default) or 'Zvn' - Zevenbergen\n"
		"\n"
		"hillshade:\n"
		"   Documentation for hillshade.\n"
		"\n"
		"Examples:\n"
		"\n"
		"GeoRsGPU slope \"c:\\gis data\\map.tif\" d:\\temp\\map_slope.tif BlockWidth=200 BlockHeight=400 Algorithm=B\n"
		"\n";
}