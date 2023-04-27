
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <dxgi1_4.h>
#include <CL/cl.h>

#include <cstdio>

int main(int argc, const char *argv[])
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    auto cudaStatus = cudaSetDevice(0);
    IDXGIFactory4* factory = nullptr;

    do
    {
        if (cudaStatus != cudaSuccess)
        {
            puts("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
            break;
        }

        if (CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory) < 0)
            break;

        // Here, we can use a warp device
        IDXGIAdapter* warpAdapter;
        if (factory->EnumWarpAdapter(__uuidof(IDXGIAdapter), (void**)&warpAdapter) < 0)
            break;

        DXGI_ADAPTER_DESC desc;
        warpAdapter->GetDesc(&desc);

        // Here, we can choose which device to use
        IDXGIAdapter1* adapter1;
        if (factory->EnumAdapters1(0, &adapter1) < 0)
            break;

        DXGI_ADAPTER_DESC1 desc1;
        adapter1->GetDesc1(&desc1);

        constexpr auto oneGB = 1024.0 * 1024.0 * 1024.0;

        wprintf(L"Current GPU: %s\n", desc1.Description);
        printf("Current available total VRAM size: %.2fGB\n", (double)desc1.DedicatedVideoMemory / oneGB);
        printf("Current available total shared VRAM size: %.2fGB\n", (double)desc1.SharedSystemMemory / oneGB);

        size_t freeVRAM = 0, totalVRAM = 0;
        cudaStatus = cudaMemGetInfo(&freeVRAM, &totalVRAM);
        if (cudaStatus == cudaSuccess)
            printf("Queried by CUDA, free VRAM: %.2fGB, total: %.2fGB\n", (double)freeVRAM / oneGB, (double)totalVRAM / oneGB);

        cl_platform_id platformID = nullptr;
        auto clState = clGetPlatformIDs(1, &platformID, nullptr);
        if (clState != CL_SUCCESS || platformID == nullptr)
            break;

        cl_device_id deviceID = nullptr;
        clState = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, nullptr);
        if (clState != CL_SUCCESS || deviceID == nullptr)
            break;

        size_t vramSize = 0;
        clState = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vramSize), &vramSize, nullptr);
        if (clState == CL_SUCCESS)
            printf("VRAM size queried by OpenCL: %.2fGB\n", (double)vramSize / oneGB);

        // Test virtual memory allocation
        constexpr auto bufferSize = 2ULL * (size_t)oneGB;

        void* vMem = VirtualAlloc(
            NULL,                       // System select
            bufferSize,                 // buffer size
            MEM_RESERVE | MEM_COMMIT,   // allocate reserved pages
            PAGE_READWRITE              // protection = read/write
        );

        if (vMem != nullptr)
        {
            constexpr auto nLoops = bufferSize / sizeof(unsigned);
            unsigned* buf = (unsigned*)vMem;
            for (size_t i = 0; i < nLoops; i++)
                buf[i] = (unsigned)i;

            VirtualFree(vMem, 0, MEM_RELEASE);
        }

    } while (false);

    if(factory != nullptr)
        factory->Release();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

