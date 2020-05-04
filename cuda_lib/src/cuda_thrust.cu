/*
 * @Author: Xu.Wang
 * @Date: 2020-05-03 17:48:57
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-04 01:36:36
 */

#include <cuda_thrust.h>

void thrustVersion() {
  printDemoTitle("CUDA Thrust Version");
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;

  cout << "Thrust version:" << major << "." << minor << endl;
  printDemoEnd();
}

void thrustHostDeviceVector() {
  printDemoTitle("CUDA Thrust Host Device Vector");

  thrust::host_vector<int> H(4);

  H[0] = 14;
  H[1] = 20;
  H[2] = 38;
  H[3] = 46;

  cout << "H has size " << H.size() << endl;

  for (int i = 0; i < H.size(); i++) {
    cout << "H[" << i << "] = " << H[i] << endl;
  }

  H.resize(2);

  cout << "H now has size " << H.size() << endl;

  // Copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;

  D[0] = 99;
  D[1] = 88;

  for (int i = 0; i < D.size(); i++) {
    cout << "D[" << i << "] = " << D[i] << endl;
  }
  printDemoEnd();
}

void thrustCopyFill() {
  printDemoTitle("CUDA Thrust Vector Copy Fill");

  thrust::device_vector<int> D(10, 1);
  thrust::fill(D.begin(), D.begin() + 7, 9);

  thrust::host_vector<int> H(D.begin(), D.begin() + 5);

  thrust::sequence(H.begin(), H.end());

  thrust::copy(H.begin(), H.end(), D.begin());

  // print D
  for (int i = 0; i < D.size(); i++)
    cout << "D[" << i << "] = " << D[i] << endl;

  printDemoEnd();
}

void thrustInitByStl() {
  printDemoTitle("CUDA Thrust Vector Init By Stl");
  std::list<int> stl_list;

  stl_list.push_back(10);
  stl_list.push_back(20);
  stl_list.push_back(30);
  stl_list.push_back(40);

  thrust::host_vector<int> H(stl_list.begin(), stl_list.end());
  thrust::device_vector<int> D(H);

  for (int i = 0; i < D.size(); i++)
    cout << "D[" << i << "] = " << D[i] << endl;

  std::vector<int> stl_vector(D.size());
  thrust::copy(D.begin(), D.end(), stl_vector.begin());

  for (int i = 0; i < stl_vector.size(); i++)
    cout << "stl_vector[" << i << "] = " << stl_vector[i] << endl;

  try {

    thrust::device_vector<int> D1(10, 3);
    for (int i = 0; i < D1.size(); i++)
      std::cout << "D1[" << i << "] = " << D1[i] << std::endl;
  } catch (thrust::system_error &e) {
    // output an error message and exit
    std::cerr << "Error accessing vector element: " << e.what() << std::endl;
    exit(-1);
  }

  printDemoEnd();
}

void ceilDivTest(int totalNum, int block_size) {
  printDemoTitle("CUDA ceilDiv Test");

  int grid_size = ceilDiv(totalNum, block_size);

  cout << "grid size:" << grid_size << endl;

  printDemoEnd();
}