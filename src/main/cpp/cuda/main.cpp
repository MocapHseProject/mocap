#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include <windows.h>


/* Class for 3-component vector storage */
class vec {
public:
  float x, y, z; // Vector coordinates

  /*  */
  explicit vec() : x(0), y(0), z(0) {}

  explicit vec(int value) : x(value), y(value), z(value) {}

  explicit vec(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};


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

  // TODO CUDA calculating

  return 0;
}
