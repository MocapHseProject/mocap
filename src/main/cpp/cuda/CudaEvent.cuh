#ifndef __cuda_cuh_
#define __cuda_cuh_

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* CUDA event class */
class CudaEvent {
  std::string NameOfEvent;
  std::vector<std::pair<int, DWORD *>> DeviceBuffers;
  std::map<std::string, int>
      OffsetsForDeviceBuffer;
  std::function<void(std::vector<std::pair<int, DWORD *>>,
                     std::map<std::string, int>)>
      Response;
  bool IsTrue;

  explicit CudaEvent(
      const std::string &Name, const std::vector<int> &BufSize,
      const std::function<void(std::vector<std::pair<int, DWORD *>>,
                               std::map<std::string, int>)> &nF,
      const std::map<std::string, int> &Offsets =
          std::map<std::string, int>()) {
    NameOfEvent = Name;
    OffsetsForDeviceBuffer = Offsets;
    if (BufSize.size() == 0) {
      std::cout << "Wrong buffers number\n";
      IsTrue = false;
      return;
    }
    DeviceBuffers.resize(BufSize.size());
    for (int i = 0; i < BufSize.size(); i++) {
      DeviceBuffers[i].first = BufSize[i];
      if (cudaMalloc((void **)&DeviceBuffers[i].second,
                     BufSize[i] * sizeof(DWORD)) != cudaSuccess)
        std::cout << "cudaMalloc failed!\n", IsTrue = false;
    }
    Response = nF;
    IsTrue = true;
  } /* End of 'CudaEvent' function */

private:
  void CopyDeviceData(std::vector<DWORD *> &HostBuffers) {
    cudaError_t Status;
    for (int i = 0; i < HostBuffers.size(); i++) {
      Status = cudaMemcpy(HostBuffers[i], DeviceBuffers[i].second,
                          DeviceBuffers[i].first * sizeof(DWORD),
                          cudaMemcpyDeviceToHost);
      if (Status != cudaSuccess) {
        std::cout << "cudaMemcpy failed!\n";
        IsTrue = false;
        return;
      }
    }
  } /* End of 'CopyDeviceData' function */

  void CopyHostData(const std::vector<DWORD *> &HostBuffers) {
    cudaError_t Status;
    for (int i = 0; i < DeviceBuffers.size(); i++) {
      Status = cudaMemcpy(DeviceBuffers[i].second, HostBuffers[i],
                          DeviceBuffers[i].first * sizeof(DWORD),
                          cudaMemcpyHostToDevice);
      if (Status != cudaSuccess) {
        std::cout << "cudaMemcpy failed!\n";
        IsTrue = false;
        return;
      }
    }
  } /* End of 'CopyDeviceData' function */

}; /* End of 'CudaEvent' class */

#endif /* __cuda_cuh_ */
