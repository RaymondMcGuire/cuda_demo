/*
 * @Author: Xu.Wang
 * @Date: 2020-04-25 03:12:11
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:29:56
 */
#include <cuda_common.h>

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

void printDemoTitle(string demo_name) {
  cout << "----------------------" << endl;
  cout << demo_name << std::endl;
}

void printDemoEnd() { cout << "----------------------" << endl; }