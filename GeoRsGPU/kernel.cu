#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "gdal_priv.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void timeIt(std::string operationName, std::function<void()> func)
{
	auto startTime = std::chrono::high_resolution_clock::now();
	func();
	auto endTime = std::chrono::high_resolution_clock::now();


	int totalMilliseconds = std::chrono::
		duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

	std::cout << "[" << operationName << "]: " << totalMilliseconds << "ms" << std::endl;
}

void executeCuda(std::function<cudaError_t()> cudaFunc)
{
	cudaError_t code = cudaFunc();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR: %d %s\n", code, cudaGetErrorString(code));
		exit(code);
	}
}

void checkCuda()
{
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR: %d %s\n", code, cudaGetErrorString(code));
		exit(code);
	}
}
void exitProgram(const char* const message)
{
	fprintf(stderr, "\n");
	fprintf(stderr, message);
	fprintf(stderr, "\n");
	exit(1);
}

void readInputFromFile(
	// Input:
	const char* const filePath,
	// Output:
	float*& inputData, double geoTransform[], int& nLineSize, int& nNumberOfLines,
	GDALDriver*& poDriver)
{
	GDALAllRegister();

	GDALDataset* poDataset = (GDALDataset*)GDALOpen(filePath, GA_ReadOnly);
	if (poDataset == NULL)
	{
		exitProgram("ERROR: Failed to open GDAL dataset.");
	}

	poDriver = GetGDALDriverManager()->GetDriverByName(poDataset->GetDriverName());
	if (poDriver == NULL)
	{
		exitProgram("ERROR: Failed to identify the GDAL driver.");
	}

	// In the particular, but common, case of a "north up" image without any rotation or shearing, 
	// the georeferencing transform takes the following form :
	// 0 -> top left x
	// 1 -> w-e pixel resolution
	// 2 -> 0
	// 3 -> top left y
	// 4 -> 0
	// 5 -> n-s pixel resolution (negative value)
	poDataset->GetGeoTransform(geoTransform);

	// We assume that the file has only one raster band
	// (we always use the first one).
	GDALRasterBand* poRasterBand = poDataset->GetRasterBand(1);
	nLineSize = poRasterBand->GetXSize();
	nNumberOfLines = poRasterBand->GetYSize();

	// Allocate pinned host memory (faster CUDA transfer performance)
	executeCuda([&]() { return cudaMallocHost((void**)&inputData, nLineSize * nNumberOfLines * sizeof(float)); });

	// Read data from file
	poRasterBand->RasterIO(GDALRWFlag::GF_Read, 0, 0, nLineSize, nNumberOfLines, inputData, nLineSize, nNumberOfLines, GDALDataType::GDT_Float32, 0, 0);

	GDALClose(poDataset);
}


int main()
{
	/*** Load input data from DEM file => inputData, geoTransform, nLineSize, nNumberOfLines, fCellSizeX, fCellSizeY, poDriver ***/
	double geoTransform[6];
	int nLineSize;
	int nNumberOfLines;
	float* inputData;
	GDALDriver *poDriver;

	timeIt("Load Data", [&]() {
		readInputFromFile(
			// input:
			"d:\\GeoGPUTeste\\Data\\dem9112.tif", // input file path
					 // output:
			inputData, geoTransform, nLineSize, nNumberOfLines, poDriver);
	});

	float fCellSizeX = (float)geoTransform[1];
	float fCellSizeY = -(float)geoTransform[5];

	std::cout << "Cell size: " << fCellSizeX << ", " << fCellSizeY << "; Raster size: " << nNumberOfLines << " rows x " << nLineSize << " columns" << std::endl;

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
