/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:06:02 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-24 03:24:52
 */
 #include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <crt_demo1.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

void crtDemo1(int width, int height, int blockX, int blockY)
{   
    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << blockX << "x" << blockY << " blocks.\n";

    int channel = 4;
    int num_pixels = height*width;
    size_t fb_size = 3*num_pixels*sizeof(float);

    // allocate FB
    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(width/blockX+1,height/blockY+1);
    dim3 threads(blockX,blockY);
    render<<<blocks, threads>>>(fb, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";

    unsigned char *data = new unsigned char[width * height * channel];
    for (int j = 0; j <height; j++) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j*3*width + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            int idx = (height-1-j) * width * channel + i * channel;
			data[idx + 0] = (unsigned char)ir;
			data[idx + 1] = (unsigned char)ig ;
			data[idx + 2] = (unsigned char)ib ;
			data[idx + 3] = (unsigned char)255 ;
        }
    }

    std::cout << "write png to file!" << std::endl;
	stbi_write_png("crt_1.png", width, height, channel, data, width * 4);
    stbi_image_free(data);
    
    checkCudaErrors(cudaFree(fb));
}
