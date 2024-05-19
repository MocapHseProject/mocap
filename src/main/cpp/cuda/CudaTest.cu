#include "CudaTest.cuh"

/* CUDA calculation for a picture filtering function.
 * ARGUMENTS:
 *   - source buffer for CUDA (texture data):
 *       DWORD *Tex_D;
 *   - width and height of texture:
 *       const int w, h;
 * RETURS: None.
 */
__global__ void RunCuda_Blur(DWORD *Tex_D, const int w, const int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= w * h)
    return;

  float Mask[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1}, sum = 0;
  d_vec color = d_vec(0);

  for (int k = -1; k <= 1; k++)
    for (int j = -1; j <= 1; j++)
      color += (V_cast(Tex_D[i + w * k + j]) * Mask[3 * (k + 1) + (j + 1)]),
          sum += Mask[3 * (k + 1) + (j + 1)];
  color /= sum;
  // color.x = d_min(d_max(color.x, 0), 1);
  // color.y = d_min(d_max(color.y, 0), 1);
  // color.z = d_min(d_max(color.z, 0), 1);
  __syncthreads();
  Tex_D[i] = color.D_cast();
} /* End of 'RunCuda_Blur' function */

/* CUDA calculation for a picture filtering function.
 * ARGUMENTS:
 *   - source buffer for CUDA (texture data):
 *       DWORD *Tex_D;
 *   - source buffer for CUDA (colors data):
 *       DWORD *Data_D;
 *   - texture size:
 *       const int size_0;
 *   - width and height of texture:
 *       const int w, h;
 *   - number of ideal colors:
 *       const int n;
 * RETURS: None.
 */
__global__ void RunCuda_RGB(DWORD *Tex_D, DWORD *Data_D, const int size_0,
                            const int w, const int h, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size_0)
    return;

  d_vec V1 = V_cast(Tex_D[i]), V = V1, col = V / 4;

  float R = 0.15, cosa, rad;
  float brightness = 0.2126 * V.x + 0.7152 * V.y + 0.0722 * V.z;

  for (int j = 0; j < n; j++) {
    d_vec a = V_cast(Data_D[j]).Normalize();
    V1.Normalize();
    cosa = V1 & a;
    rad = !(V1 - a * cosa);
    if (rad <= R && cosa > 0.993 && brightness > 0.31) {
      col = V;
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 0]), 1);
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 1]),
                i - w * (i / w));
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 2]), i / w + 1);
    }
  }
  Tex_D[i] = col.D_cast();
} /* End of 'RunCuda_RGB' function */

/* CUDA calculation for a picture filtering function.
 * ARGUMENTS:
 *   - source buffer for CUDA (texture data):
 *       DWORD *Tex_D;
 *   - source buffer for CUDA (colors data):
 *       DWORD *Data_D;
 *   - texture size:
 *       const int size_0;
 *   - width and height of texture:
 *       const int w, h;
 *   - number of ideal colors:
 *       const int n;
 * RETURS: None.
 */
__global__ void RunCuda_HSV(DWORD *Tex_D, DWORD *Data_D, const int size_0,
                            const int w, const int h, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size_0)
    return;

  d_vec V = V_cast(Tex_D[i]), col = V / 4, res = RGB2HSV(V);
  int b = int(res.x / 30);

  for (int j = 0; j < n; j++) {
    d_vec a = RGB2HSV(Data_D[j]);
    int a_b = int(a.x / 30);
    if (abs(res.y - a.y) < 0.3 && abs(res.x - a.x) < 10) {
      col = V;
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 0]), 1);
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 1]),
                i - w * (i / w));
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 2]), i / w + 1);
    }
  }
  Tex_D[i] = col.D_cast();
} /* End of 'RunCuda_HSV' function */

/* CUDA calculation for a picture filtering function.
 * ARGUMENTS:
 *   - source buffer for CUDA (texture data):
 *       DWORD *Tex_D;
 *   - source buffer for CUDA (colors data):
 *       DWORD *Data_D;
 *   - texture size:
 *       const int size_0;
 *   - width and height of texture:
 *       const int w, h;
 *   - number of ideal colors:
 *       const int n;
 * RETURS: None.
 */
__global__ void RunCuda_LAB(DWORD *Tex_D, DWORD *Data_D, const int size_0,
                            const int w, const int h, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size_0)
    return;

  d_vec V = V_cast(Tex_D[i]), col = V / 4,
        white_point(0.9505, 1,
                    1.0889), // default white point color in XYZ (D65)
      res = RGB2LAB(V, white_point);

  for (int j = 0; j < n; j++) {
    d_vec a = RGB2LAB(Data_D[j], white_point);

    if ((pow((a.y - res.y), 2) + pow((a.z - res.z), 2)) < 300) {
      col = V;
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 0]), 1);
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 1]),
                i - w * (i / w));
      atomicAdd(reinterpret_cast<int *>(&Data_D[n + 3 * j + 2]), i / w + 1);
    }
  }
  Tex_D[i] = col.D_cast();
} /*End of 'RunCuda_LAB' function */

/* CUDA calculation for joint's position finding function.
 * ARGUMENTS:
 *   - source buffer for CUDA (lines data: normal, one point on the plane):
 *       DWORD *Data_D;
 *   - number of conncted devices:
 *       const int NumOfDevices;
 *   - number of ideal colors:
 *       const int n;
 * RETURS: None.
 */
__global__ void RunCuda_2(DWORD *Data_D, const int NumOfDevices, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= NumOfDevices)
    return;
} /* End of 'RunCuda_2' function */

/* CUDA response for textures bluring function.
 * ARGUMENTS:
 *   - vector of device biffers:
 *       std::vector<std::pair<int, DWORD *>> DeviceBuffers;
 *   - offsets for device buffers:
 *       std::map<std::string, int> Offsets;
 * RETURNS: None.
 */
void BlurResponse(std::vector<std::pair<int, DWORD *>> DeviceBuffers,
                  std::map<std::string, int> Offsets) {
  int x_threads = DeviceBuffers[0].first / 1024 + 1,
      y_threads = h_min(1024, DeviceBuffers[0].first);
  RunCuda_Blur<<<x_threads, y_threads>>>(DeviceBuffers[0].second, Offsets["w"],
                                         Offsets["h"]);
} /* End of 'Blur' function */

/* CUDA response for textures parcing function.
 * ARGUMENTS:
 *   - vector of device biffers:
 *       std::vector<std::pair<int, DWORD *>> DeviceBuffers;
 *   - offsets for device buffers:
 *       std::map<std::string, int> Offsets;
 * RETURNS: None.
 */
void TexturesResponse(std::vector<std::pair<int, DWORD *>> DeviceBuffers,
                      std::map<std::string, int> Offsets) {
  int x_threads = DeviceBuffers[0].first / 1024 + 1,
      y_threads = h_min(1024, DeviceBuffers[0].first);
  if (Offsets["color"] == 0)
    RunCuda_RGB<<<x_threads, y_threads>>>(
        DeviceBuffers[0].second, DeviceBuffers[1].second,
        DeviceBuffers[0].first, Offsets["w"], Offsets["h"], Offsets["n"]);
  if (Offsets["color"] == 1)
    RunCuda_HSV<<<x_threads, y_threads>>>(
        DeviceBuffers[0].second, DeviceBuffers[1].second,
        DeviceBuffers[0].first, Offsets["w"], Offsets["h"], Offsets["n"]);
  if (Offsets["color"] == 2)
    RunCuda_LAB<<<x_threads, y_threads>>>(
        DeviceBuffers[0].second, DeviceBuffers[1].second,
        DeviceBuffers[0].first, Offsets["w"], Offsets["h"], Offsets["n"]);
} /* End of 'TexturesResponse' function */

/* CUDA response for textures parcing function.
 * ARGUMENTS:
 *   - vector of device biffers:
 *       std::vector<std::pair<int, DWORD *>> DeviceBuffers;
 *   - offsets for device buffers:
 *       std::map<std::string, int> Offsets;
 * RETURNS: None.
 */
void JointsResponse(std::vector<std::pair<int, DWORD *>> DeviceBuffers,
                    std::map<std::string, int> Offsets) {
  if (Offsets["NumOfDevices"] < 2) {
    std::cout << "Connect more than one device for this function calling\n";
    return;
  }
  int x_threads = DeviceBuffers[0].first / 1024 + 1,
      y_threads = h_min(1024, DeviceBuffers[0].first);
  RunCuda_2<<<x_threads, y_threads>>>(DeviceBuffers[0].second,
                                      Offsets["NumOfDevices"], Offsets["n"]);
} /* End of 'JointsResponse' function */
