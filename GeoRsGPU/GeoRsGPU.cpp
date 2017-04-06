#include <iostream>
#include <vector>
#include <string>
#include <args.hxx>

enum RasterCommand
{
	Slope,
	Hillshade
};

int main(int argc, char *argv[])
{

	std::cerr << "GeoRsGPU - v1.0.100" << std::endl << std::endl;

	std::unordered_map<std::string, RasterCommand> map{
		{ "slope", RasterCommand::Slope },
		{ "hillshade", RasterCommand::Hillshade } };

	args::ArgumentParser parser(
		"GeoRsGPU - GPU accelerated raster processing (requires NVIDIA CUDA)",
		"\n"
		"Command-specific documentation\n"
		"\n"
		"\n"
		"slope:\n"
		"   Documentation for slope.\n"
		"\n"
		"hillshade:\n"
		"   Documentation for hillshade.\n"
		"\n"
		"Examples:\n"
		"\n"
		"GeoRsGPU slope \"c:\\gis data\\map.tif\" d:\\temp\\map_slope.tif BlockWidth=200 BlockHeight=400 Algorithm=B\n"
		"\n"
	);
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
		RasterCommand cmd = posCommand.Get();
		std::cout << "Command:" << cmd << std::endl;
		std::cout << "Input:" << posInputFile.Get() << std::endl;
		std::cout << "Output:" << posOutputFile.Get() << std::endl;
		std::cout << "Optional parameters:" << std::endl;
		std::vector<std::string> v = parameters.Get();
		for (int i = 0; i < v.size(); i++)
		{
			std::cout << "   " << v[i] << std::endl;
		}
	}
	catch (args::Help)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError e)
	{
		std::cerr << "ERROR parsing arguments: " << e.what()
			<< std::endl
			<< "See 'GeoRsGPU --help' for documentation"
			<< std::endl;

		return 1;
	}
	catch (args::ValidationError e)
	{
		std::cerr << "ERROR parsing arguments: " << e.what()
			<< std::endl << std::endl
			<< "See 'GeoRsGPU --help' for documentation"
			<< std::endl;

		return 1;
	}
	return 0;
}