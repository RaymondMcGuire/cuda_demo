/*
 * @Author: Xu.Wang
 * @Date: 2020-05-04 00:37:43
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-04 01:21:25
 */
#include <cuda_math.h>
#include <vector3.h>
int mathVec3() {
  printDemoTitle("CUDA Math Vector3");

  thrust::device_vector<cuda_math::Vector3> pos(
      5, cuda_math::Vector3(1.0f, 2.0f, 3.0f));

  for (int i = 0; i < pos.size(); i++) {
    cout << "pos[" << i << "] = " << ((cuda_math::Vector3)pos[i]).x() << ","
         << ((cuda_math::Vector3)pos[i]).y() << ","
         << ((cuda_math::Vector3)pos[i]).z() << endl;
  }

  printDemoEnd();
  return 0;
}