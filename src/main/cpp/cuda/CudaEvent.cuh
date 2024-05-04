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
  std::string NameOfEvent; // Name of the current CUDA event
  std::vector<std::pair<int, DWORD *>> DeviceBuffers; // Device buffers for CUDA
  std::map<std::string, int>
      OffsetsForDeviceBuffer; // Offsets in CUDA device buffers for response
                              // function
  std::function<void(std::vector<std::pair<int, DWORD *>>,
                     std::map<std::string, int>)>
      Response; // Response function for CUDA calculating
  bool IsTrue;  // If this flag is false there was some mistake in current CUDA
                // event

  /* Class constructor function.
   * ARGUMENTS:
   *   - name of CUDA event:
   *       const std::string &name;
   *   - size of device buffers for CUDA:
   *       const std::vector<int> &BufSize;
   *   - response function for CUDA:
   *       const std::function<void(std::vector<std::pair<int, DWORD *>>,
   * std::map<std::string, int>)> &nF;
   */
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

public:
  /* Get event name function.
   * ARGUMENTS: None.
   * RETURNS:
   *   (const std::string &) - name of the event.
   */
  const std::string &GetEventName() {
    return NameOfEvent;
  } /* End of 'GetEventName' function */

  void SetOffset(const std::string &FieldName, const int Value) {
    OffsetsForDeviceBuffer[FieldName] = Value;
  } /* End of 'SetOffset' function */

  /* Release CUDA event function.
   * ARGUMENTS:
   *   - offsets in buffers:
   *       const std::map<std::string, int> &Src;
   * RETURNS: None.
   */
  void ReleaseEvent(std::vector<DWORD *> &HostBuffers) {
    if (HostBuffers.size() != DeviceBuffers.size())
      std::cout << "Sizes of host and device buffers are not equal\n",
          IsTrue = false;
    if (IsTrue) {
      CopyHostData(HostBuffers);
      Response(DeviceBuffers, OffsetsForDeviceBuffer);
      CopyDeviceData(HostBuffers);
    } else
      CloseEvent();
  } /* End of 'ReleaseEvent' function */

  /* Change response function.
   * ARGUMENTS:
   *   - new response function:
   *       const std::function<void(std::vector<std::pair<int, DWORD *>>,
   * std::map<std::string, int>)> &nF; RETURNS: None.
   */
  void ChangeResponseFunction(
      const std::function<void(std::vector<std::pair<int, DWORD *>>,
                               std::map<std::string, int>)> &nF) {
    Response = nF;
  } /* End of 'ChangeResponseFunction' function */

  /* Resize certain device buffer function.
   * ARGUMENTS:
   *   - buffer index:
   *       const int BufferIndex;
   * RETURNS: None.
   */
  void ResizeBuffer(const int BufferIndex) {
    if (DeviceBuffers[BufferIndex].second != nullptr)
      cudaFree(DeviceBuffers[BufferIndex].second);
    if (cudaMalloc((void **)&DeviceBuffers[BufferIndex].second,
                   DeviceBuffers[BufferIndex].first * sizeof(DWORD)) !=
        cudaSuccess)
      std::cout << "cudaMalloc failed!\n", IsTrue = false;
  } /* End of 'ResizeBuffer' function */

private:
  /* Copy data from device to host buffers function.
   * ARGUMENTS:
   *   - host buffers vector:
   *       std::vector<DWORD *> &HostBuffers;
   * RETURNS: None.
   */
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

  /* Copy data from host to device buffers function.
   * ARGUMENTS: None.
   * RETURNS: None.
   */
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

  /* Close event function.
   * ARGUMENTS: None.
   * RETURNS: None.
   */
  void CloseEvent() {
    if (cudaDeviceReset() != cudaSuccess)
      std::cout << "cudaDeviceReset failed!\n";
    if (!DeviceBuffers.empty())
      for (auto &b : DeviceBuffers)
        if (b.second != nullptr)
          cudaFree(b.second);
  } /* End of 'CloseEvent' function */

  /* Class destructor function */
  ~CudaEvent() { CloseEvent(); } /* End of '~CudaEvent' function */
}; /* End of 'CudaEvent' class */

#endif /* __cuda_cuh_ */
