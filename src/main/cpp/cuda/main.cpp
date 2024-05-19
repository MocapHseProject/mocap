#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include <windows.h>

#include "CudaEvent.cuh"
#include "CudaTest.cuh"

std::vector<CudaEvent> CudaEventsList;

/* Add CUDA event function.
 * ARGUMENTS:
 *   - name of CUDA event:
 *       const std::string &name;
 *   - size of device buffers for CUDA:
 *       const std::vector<int> &BufSize;
 *   - response function for CUDA:
 *       const std::function<void(std::map<std::string, int>)> &nF;
 * RETURNS:
 *   (CudaEvent *) - CUDA event pointer.
 */
CudaEvent *
CudaEventAdd(const std::string &Name, const std::vector<int> &BufSize,
             const std::function<void(std::vector<std::pair<int, DWORD *>>,
                                      std::map<std::string, int>)> &nF,
             const std::map<std::string, int> &Offets) {
  CudaEventsList.push_back(new CudaEvent(Name, BufSize, nF, Offets));
  return CudaEventsList.back();
} /* End of 'CudaEventAdd' function */

/* Get CUDA event by name function.
 * ARGUMENTS:
 *   - cuda event's name:
 *       const std::string &Name;
 * RETURNS:
 *   (CudaEvent *) - cuda event pointer.
 */
CudaEvent *GetCudaEventByName(const std::string &Name) {
  for (auto &e : CudaEventsList)
    if (e->GetEventName() == Name)
      return e;
  return nullptr;
} /* End of 'GetCudaEventByName' function */

/* Class for 3-component vector storage */
class vec {
public:
  float x, y, z; // Vector coordinates

  /*  */
  explicit vec() : x(0), y(0), z(0) {}

  explicit vec(int value) : x(value), y(value), z(value) {}

  explicit vec(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

/* Convert vector to dword number for color (abgr format) function.
 * ARGUMENTS:
 *   - color range:
 *       const int N;
 * RETURNS:
 *   (DWORD) - color from vector;
 */
static DWORD D(const vec &V, const int N = 255) {
  return (int(V.z * N) << 0) | (int(V.y * N) << 8) | (int(V.x * N) << 16) |
         0xFF000000;
} /* End of 'D' function */

void init(int w, int h, int n, int NumOfDevices = 1) {
  // Initialize CUDA events
  std::function Response = TexturesResponse;
  CudaEventAdd(
      "TexturesParser", std::vector<int>({w * h, 4 * n}), Response,
      std::map<std::string, int>({{"w", w}, {"h", h}, {"n", n}, {"color", 0}}));

  // Structure of data buffer: (k * n) lines (normal + 3 coordinates of basic
  // line point); n output points of each joint position
  Response = JointsResponse;
  CudaEventAdd(
      "JointsPositions", std::vector<int>({NumOfDevices * n * (1 + 3) + 3 * n}),
      Response,
      std::map<std::string, int>({{"NumOfDevices", NumOfDevices}, {"n", n}}));

  // Bluring texture CUDA event
  Response = BlurResponse;
  CudaEventAdd("TextureBlur", std::vector<int>({w * h}), Response,
               std::map<std::string, int>({{"w", w}, {"h", h}}));
}

int main(void) {
  // Constants
  const int n = 3;
  const int w = 480;
  const int h = 640;

  // Init CUDA, data array and plane primitives
  const vec ideal[n] = {vec(0.92, 0.85, 0.3), vec(0.22, 0.78, 0.42),
                        vec(0.8, 0.21, 0.19)};

  // Create CUDA data array
  DWORD *TextureData = new DWORD[w * h];
  DWORD *Data = new DWORD[n + n * 3];
  for (int i = 0; i < n; i++) {
    // Init its colors params
    Data[i] = D(ideal[i]);
    Data[n + 3 * i + 0] = 0;
    Data[n + 3 * i + 1] = 0;
    Data[n + 3 * i + 2] = 0;
  }

  // CUDA calculating
  std::vector<DWORD *> CUDA_tmp = {TextureData};
  GetCudaEventByName("TextureBlur")->ReleaseEvent(CUDA_tmp);
  CUDA_tmp = {TextureData, Data};
  GetCudaEventByName("TexturesParser")->ReleaseEvent(CUDA_tmp);

  return 0;
}