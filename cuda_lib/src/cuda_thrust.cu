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
  for (int i = 0; i < D.size(); i++)
    cout << "D[" << i << "] = " << D[i] << endl;
  thrust::fill(D.begin(), D.begin() + 7, 9);
  for (int i = 0; i < D.size(); i++)
    cout << "D[" << i << "] = " << D[i] << endl;

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

void thrustTransform() {
  printDemoTitle("CUDA thrust transform");
  const int N = 5;
  double x[N] = {0, 1, 2, 3, 4};
  double y[N] = {1, 2, 3, 4, 5};

  double length[N];

  thrust::device_vector<double> xVector(x, x + N);
  thrust::device_vector<double> yVector(y, y + N);
  thrust::device_vector<double> lengthVector(N);

  thrust::transform(xVector.begin(), xVector.begin() + N, yVector.begin(),
                    lengthVector.begin(), CalcSqrtDim2());
  thrust::copy_n(lengthVector.begin(), N, length);

  for (int i = 0; i < N; i++) {
    cout << i << ": " << length[i] << endl;
  }

  printDemoEnd();
}

void thrustTransformDevicePtr() {
  printDemoTitle("CUDA thrust transform devicePtr");
  const int N = 5;

  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));

  for (size_t i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  int *aa;
  int *bb;
  float *cc;
  cudaMalloc((void **)&aa, sizeof(int) * N);
  cudaMalloc((void **)&bb, sizeof(int) * N);
  cudaMalloc((void **)&cc, sizeof(float) * N);

  cudaMemcpy(aa, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(bb, b, N * sizeof(int), cudaMemcpyHostToDevice);

  thrust::device_ptr<int> pa(const_cast<int *>(aa));
  thrust::device_ptr<int> pb(const_cast<int *>(bb));
  thrust::device_ptr<float> pc(cc);
  thrust::transform(pa, pa + N,      // a for input
                    pb,              // b for input
                    pc,              // c for output
                    CalcSqrtDim2()); // z = sqrt(x^2 + y^2)

  cudaDeviceSynchronize();

  for (size_t i = 0; i < N; i++) {
    cout << "c[" << i << "]:" << pc[i] << endl;
  }
  printDemoEnd();
}

void thrustSortByKey() {
  printDemoTitle("CUDA thrust sortbykey");
  const int N = 6;
  int keys[N] = {1, 4, 2, 8, 5, 7};
  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

  int *ptrKey;
  char *ptrVal;

  cudaMalloc((void **)&ptrKey, sizeof(int) * N);
  cudaMalloc((void **)&ptrVal, sizeof(char) * N);
  cudaMemcpy(ptrKey, keys, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ptrVal, values, N * sizeof(char), cudaMemcpyHostToDevice);

  thrust::device_ptr<int> tKeys(ptrKey);
  thrust::device_ptr<char> tVals(ptrVal);
  thrust::sort_by_key(tKeys, tKeys + N, tVals);

  thrust::device_ptr<int> ttKeys(ptrKey);
  thrust::device_ptr<char> ttVals(ptrVal);

  for (size_t i = 0; i < N; i++) {
    //cout << "keys[" << i << "]:" << keys[i] << endl;
    //cout << "values[" << i << "]:" << values[i] << endl;

    cout << "tkeys[" << i << "]:" << tKeys[i] << endl;
    cout << "tvalues[" << i << "]:" << tVals[i] << endl;

    cout << "ttKeys[" << i << "]:" << ttKeys[i] << endl;
    cout << "ttVals[" << i << "]:" << ttVals[i] << endl;
  }
  printDemoEnd();
}


void thrustMerge() {
    printDemoTitle("CUDA thrust merge(merge two array and sort)");
    int A1[6] = {1, 3, 5, 7, 9, 11};
    int A2[7] = {1, 1, 2, 3, 5,  8, 13};

    int *ptrA1;
    int *ptrA2;
    int *result;
  
    cudaMalloc((void **)&ptrA1, sizeof(int) * 6);
    cudaMalloc((void **)&ptrA2, sizeof(int) * 7);
    cudaMalloc((void **)&result, sizeof(int) * 13);

    cudaMemcpy(ptrA1, A1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptrA2, A2, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
    thrust::device_ptr<int> dA1(ptrA1);
    thrust::device_ptr<int> dA2(ptrA2);
    thrust::device_ptr<int> dresult(result);

    thrust::merge(dA1, dA1 + 6, dA2, dA2 + 7, dresult);
  
    for (size_t i = 0; i < 13; i++) {

      cout << "dresult[" << i << "]:" << dresult[i] << endl;
    }
    printDemoEnd();
  }


  void thrustMergeTwoArrays() {
    printDemoTitle("CUDA Thrust Merge Two Arrays");
  
    int A1[6] = {1, 3, 5, 7, 9, 11};
    int A2[7] = {1, 1, 2, 3, 5,  8, 13};

    int *ptrA1;
    int *ptrA2;
    int *result;
  
    cudaMalloc((void **)&ptrA1, sizeof(int) * 6);
    cudaMalloc((void **)&ptrA2, sizeof(int) * 7);
    cudaMalloc((void **)&result, sizeof(int) * 13);

    cudaMemcpy(ptrA1, A1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptrA2, A2, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
    thrust::device_ptr<int> dA1(ptrA1);
    thrust::device_ptr<int> dA2(ptrA2);
    thrust::device_ptr<int> dresult(result);

    thrust::copy(dA1, dA1+6, result);
    thrust::copy(dA2, dA2+7, result+6);
    for (size_t i = 0; i < 13; i++) {

        cout << "dresult[" << i << "]:" << dresult[i] << endl;
      }
    printDemoEnd();
  }