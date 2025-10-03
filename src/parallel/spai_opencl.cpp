#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

std::string loadFile(const std::string &path) {
    std::ifstream f(path);
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

int main() {
    // Problem size
    const int n = 4;
    std::vector<float> A = {
        4,1,0,0,
        1,3,1,0,
        0,1,2,1,
        0,0,1,2
    };
    std::vector<float> M(n*n, 0.0f);

    // Get platforms
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);

    // Get device (GPU)
    cl_device_id device;
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Load kernel
    std::string kernelSrc = loadFile("spai.cl");
    const char* src = kernelSrc.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "spai_full", nullptr);

    // Create buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*A.size(), A.data(), nullptr);
    cl_mem bufM = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*M.size(), nullptr, nullptr);

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufM);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    // Launch kernel: one work-item per column
    size_t global_work_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);

    // Read back result
    clEnqueueReadBuffer(queue, bufM, CL_TRUE, 0, sizeof(float)*M.size(), M.data(), 0, nullptr, nullptr);

    // Print result
    std::cout << "Full SPAI (dense) result:\n";
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            std::cout << M[i*n+j] << " ";
        std::cout << "\n";
    }

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufM);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
