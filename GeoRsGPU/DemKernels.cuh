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

#ifndef DEM_KERNELS_H
#define DEM_KERNELS_H

#include "cuda_runtime.h";

namespace GeoRsGpu {

	__device__ const float RadDegree = 57.29578f;

	struct KernelSlopeZevenbergen
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float dZdY = (b - h) / (2 * cellSizeY);
			float dZdX = (f - d) / (2 * cellSizeX);
			return (sqrt(pow(dZdX, 2) + pow(dZdY, 2))) * RadDegree;
		}
	};

	struct KernelSlopeBurruogh {
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{

			float dZdY = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellSizeY);
			float dZdX = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellSizeX);
			return atan(sqrt(pow(dZdX, 2) + pow(dZdY, 2))) * RadDegree;
		}
	};

	struct KernelHillshade
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float dZdY = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellSizeY);
			float dZdX = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellSizeX);

			float A = ((a + c + g + i) / 4 - (b + d + f + h) / 2 + e) / pow(cellSizeX, 4);
			float B = ((a + c - g - i) / 4 - (b - h) / 2) / pow(cellSizeX, 3);
			float C = ((-a + c - g + i) / 4 + (d - f) / 2) / pow(cellSizeX, 3);
			float D = ((d + f) / 2 - e) / pow(cellSizeX, 2);
			float E = ((b + h) / 2 - e) / pow(cellSizeX, 2);
			float F = (-a + c + g - i) / 4 * pow(cellSizeX, 2);
			float G = (-d + f) / 2 * cellSizeX;
			float H = (b - h) / 2 * cellSizeX;

			float aspect = atan2f(-H, -G);
			if (aspect < 0)
				aspect = 90.0 - aspect;
			else if (aspect > 90.0)
				aspect = 360.0 - aspect + 90.0;
			else
				aspect = 90.0 - aspect;

			float azimuthMath = 360.0 - 310 + 90;
			float zenithRad = 45 * 3.1438571429 / 180.0;
			float slope = atan(sqrt(pow(dZdX, 2) + pow(dZdY, 2)));
			float azimutRad = azimuthMath * 3.1438571429 / 180.0;

			return 255.0 * ((cos(zenithRad) * cos(slope)) +
				(sin(zenithRad) * sin(slope) * cos(azimutRad - aspect)));
		}
	};

	struct KernelAspect
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float A = ((a + c + g + i) / 4 - (b + d + f + h) / 2 + e) / pow(cellSizeX, 4);
			float B = ((a + c - g - i) / 4 - (b - h) / 2) / pow(cellSizeX, 3);
			float C = ((-a + c - g + i) / 4 + (d - f) / 2) / pow(cellSizeX, 3);
			float D = ((d + f) / 2 - e) / pow(cellSizeX, 2);
			float E = ((b + h) / 2 - e) / pow(cellSizeX, 2);
			float F = (-a + c + g - i) / 4 * pow(cellSizeX, 2);
			float G = (-d + f) / 2 * cellSizeX;
			float H = (b - h) / 2 * cellSizeX;

			float aspect = atan2f(-H, -G) * RadDegree;
			if (aspect < 0)
				aspect = 90.0 - aspect;
			else if (aspect > 90.0)
				aspect = 360.0 - aspect + 90.0;
			else
				aspect = 90.0 - aspect;

			return aspect;
		}
	};

	struct KernelTotalCurvature {
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{

			float A = ((a + c + g + i) / 4 - (b + d + f + h) / 2 + e) / pow(cellSizeX, 4);
			float B = ((a + c - g - i) / 4 - (b - h) / 2) / pow(cellSizeX, 3);
			float C = ((-a + c - g + i) / 4 + (d - f) / 2) / pow(cellSizeX, 3);
			float D = ((d + f) / 2 - e) / pow(cellSizeX, 2);
			float E = ((b + h) / 2 - e) / pow(cellSizeX, 2);
			float F = (-a + c + g - i) / 4 * pow(cellSizeX, 2);
			float G = (-d + f) / 2 * cellSizeX;
			float H = (b - h) / 2 * cellSizeX;
			float I = e;
			return -2 * (D + E) * 100;
		}
	};

	struct KernelProfileCurvature {
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{

			float A = ((a + c + g + i) / 4 - (b + d + f + h) / 2 + e) / pow(cellSizeX, 4);
			float B = ((a + c - g - i) / 4 - (b - h) / 2) / pow(cellSizeX, 3);
			float C = ((-a + c - g + i) / 4 + (d - f) / 2) / pow(cellSizeX, 3);
			float D = ((d + f) / 2 - e) / pow(cellSizeX, 2);
			float E = ((b + h) / 2 - e) / pow(cellSizeX, 2);
			float F = (-a + c + g - i) / 4 * pow(cellSizeX, 2);
			float G = (-d + f) / 2 * cellSizeX;
			float H = (b - h) / 2 * cellSizeX;
			float I = e;
			return -2 * (D*pow(G, 2) + E * pow(H, 2) + F*G*H) / (pow(G, 2) + pow(H, 2));
		}
	};

	struct KernelPlanCurvature {
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{

			float A = ((a + c + g + i) / 4 - (b + d + f + h) / 2 + e) / pow(cellSizeX, 4);
			float B = ((a + c - g - i) / 4 - (b - h) / 2) / pow(cellSizeX, 3);
			float C = ((-a + c - g + i) / 4 + (d - f) / 2) / pow(cellSizeX, 3);
			float D = ((d + f) / 2 - e) / pow(cellSizeX, 2);
			float E = ((b + h) / 2 - e) / pow(cellSizeX, 2);
			float F = (-a + c + g - i) / 4 * pow(cellSizeX, 2);
			float G = (-d + f) / 2 * cellSizeX;
			float H = (b - h) / 2 * cellSizeX;
			float I = e;

			return 2 * (D * pow(H, 2) + E * pow(G, 2) - F * G * H) / (pow(G, 2) + pow(H, 2));
		}
	};

	struct KernelTPI_EP
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float window[9] = { a, b, c, d, e, f, g , h, i };
			float counter = 0;
			for(int m = 0; m < 9; m++)
			{
				e < window[m] ? counter++ : counter;
			}
			return counter * 100 / 8.0f;
		}
	};

	struct KernelTPI_DIF
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float meanWindow = (a + b + c + d + e + f + g + h + i) / 9.0f;
			float stdevWindow = sqrt((pow(a - meanWindow, 2) + pow(b - meanWindow, 2) + pow(c - meanWindow, 2) + pow(d - meanWindow, 2) + pow(e - meanWindow, 2) + pow(f - meanWindow, 2) + pow(g - meanWindow, 2) + pow(h - meanWindow, 2) + pow(i - meanWindow, 2)) / 9.0f);
			return (e - meanWindow);
		}
	};
	struct KernelTPI_DEV
	{
		__device__ float operator()(
			const float a, const float b, const float c,
			const float d, const float e, const float f,
			const float g, const float h, const float i,
			const float cellSizeX, const float cellSizeY)
		{
			float meanWindow = (a + b + c + d + e + f + g + h + i) / 9.0f;
			float stdevWindow = sqrt((pow(a - meanWindow, 2) + pow(b - meanWindow, 2) + pow(c - meanWindow, 2) + pow(d - meanWindow, 2) + pow(e - meanWindow, 2) + pow(f - meanWindow, 2) + pow(g - meanWindow, 2) + pow(h - meanWindow, 2) + pow(i - meanWindow, 2)) / 9.0f);
			return (e - meanWindow) / stdevWindow;
		}
	};
}
#endif // !DEM_KERNELS_H