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

#ifndef GEORSGPU_RASTER_COMMAND_H
#define GEORSGPU_RASTER_COMMAND_H

namespace GeoRsGpu {

	enum class RasterCommand
	{
		Slope,
		Hillshade,
		Aspect,
		TotalCurvature,
		PlanCurvature,
		ProfileCurvature,
		TopographicPositionIndex,
		Majority,
		Minority,
		Maximum,
		Minimum,
		Median,
		Mean,
		StandardDeviation,
		Range,
		Sum,
		Variety,
		Percentile,
		DiffFromMean,
		StDevFromMean,
		ATan,
		ATanH,
		//ATan2,
		ACos,
		ACosH,
		ASin,
		ASinH,
		Tan,
		TanH,
		Cos,
		CosH,
		Sin,
		SinH,
		Abs,
		RoundUp,
		RoundDown,
		Expf,
		Exp10f,
		Exp2f,
		Lnf,
		Log10f,
		Log2f,
		Negate,
		Power,
		Square,
		SqRoot
	};
}

#endif // !GEORSGPU_RASTER_COMMAND_H
