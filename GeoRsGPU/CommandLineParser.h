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

#ifndef GEORSGPU_COMMAND_LINE_PARSER_H
#define GEORSGPU_COMMAND_LINE_PARSER_H

#include <limits>
#include <map>
#include <string>

#include "RasterCommand.h"

namespace GeoRsGpu {

	class CommandLineParser {
	public:
		CommandLineParser(int argc, char *argv[]);

		inline std::string getInputFileName() const { return m_inputFileName; }
		inline std::string getOutputFileName() const { return m_outputFileName; }
		inline RasterCommand getCommand() const { return m_command; }
		inline int getBlockHeight() const { return m_blockHeight; }
		inline int getBlockWidth() const { return m_blockWidth; }
		
		inline bool parameterExists(std::string paramName) { 
			return m_parameters.find(paramName) != m_parameters.end(); }		
		inline std::string getStringParameter(std::string paramName) { 
			return m_parameters[paramName]; }
		bool isIntParameterValid(std::string paramName, 
			int minValue = std::numeric_limits<int>::min());
		inline int getIntParameter(std::string paramName) {
			return std::stoi(m_parameters[paramName]);
		}
		inline int getFloatParameter(std::string paramName) {
			return std::stof(m_parameters[paramName]);
		}
		inline int getFloatParameter(std::string paramName, float defaultValue) {
			if (parameterExists(paramName))
				return getFloatParameter(paramName);
			else
				return defaultValue;
		}

		inline bool isValid() const { return m_isValid; }

	private:
		std::string m_inputFileName;
		std::string m_outputFileName;
		RasterCommand m_command;
		int m_blockHeight;
		int m_blockWidth;
		
		std::map<std::string, std::string> m_parameters;

		bool m_isValid;

		inline std::string trim(const std::string& str) const
		{
			size_t first = str.find_first_not_of(' ');
			if (std::string::npos == first)
			{
				return str;
			}
			size_t last = str.find_last_not_of(' ');
			return str.substr(first, (last - first + 1));
		}

		std::string getDescription();
	};
	
}

#endif // !GEORSGPU_COMMAND_LINE_PARSER_H