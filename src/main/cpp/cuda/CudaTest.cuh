#ifndef __cuda_test_cuh_
#define __cuda_test_cuh_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <windows.h>

/* Vector class for CUDA */
class d_vec {
public:
  float x, y, z; // Vector components

  /* Default constructor. */
  __device__ d_vec(void) {} /* End of 'vec' function */

  __device__ d_vec(const float a) { x = y = z = a; } /* End of 'vec' function */

  __device__ d_vec(const float q, const float w, const float e) {
    x = q;
    y = w;
    z = e;
  } /* End of 'vec' function */

  __device__ d_vec operator-(const d_vec &V) const {
    return d_vec(x - V.x, y - V.y, z - V.z);
  } /* End of 'operator-' function */

  __device__ d_vec operator+(const d_vec &V) const {
    return d_vec(x + V.x, y + V.y, z + V.z);
  } /* End of 'operator+' function */

  __device__ void operator+=(const d_vec &V) {
    x += V.x;
    y += V.y;
    z += V.z;
  } /* End of 'operator+=' function */

  __device__ void operator-=(const d_vec &V) {
    x -= V.x;
    y -= V.y;
    z -= V.z;
  } /* End of 'operator-=' function */

  __device__ void operator/=(const float a) {
    x /= a;
    y /= a;
    z /= a;
  } /* End of 'operator/=' function */

  __device__ void operator*=(const float a) {
    x *= a;
    y *= a;
    z *= a;
  } /* End of 'operator*=' function */

  __device__ float operator&(const d_vec &V) const {
    return x * V.x + y * V.y + z * V.z;
  } /* End of 'operator&' function */

  __device__ float operator!(void) const {
    return sqrt(x * x + y * y + z * z);
  } /* End of 'operator!' function */

  __device__ d_vec operator*(const float N) const {
    return d_vec(x * N, y * N, z * N);
  } /* End of 'operator*' function */

  __device__ d_vec operator/(const float N) const {
    return d_vec(x / N, y / N, z / N);
  } /* End of 'operator/' function */

  __device__ d_vec &Normalize(void) {
    const float n(sqrt(x * x + y * y + z * z));
    if (n != 0 && n != 1) {
      x /= n;
      y /= n;
      z /= n;
    }
    return *this;
  } /* End of 'Normalize' function */

  __device__ d_vec Normalizing(void) const {
    const float l(sqrt(x * x + y * y + z * z));
    if (l != 1 && l != 0)
      return d_vec(x / l, y / l, z / l);
    return *this;
  } /* End of 'Normalizing' function */

  __device__ DWORD D_cast(int N = 255) const {
    return (int(z * N) << 0) | (int(y * N) << 8) | (int(x * N) << 16) |
           0xFF000000;
  } /* End of 'D' function */
};  /* End of 'd_vec' class */

/* Matrix 3x3 class */
class d_matr3 {
  float M[3][3]; // matrix array

public:
  __device__ d_matr3(const float A00, const float A01, const float A02,
                     const float A10, const float A11, const float A12,
                     const float A20, const float A21, const float A22)
      : M{{A00, A01, A02}, {A10, A11, A12}, {A20, A21, A22}} {
  } /* End of 'd_matr3' function */

  __device__ static float MatrDeterm2x2(const float A00, const float A01,
                                        const float A10, const float A11) {
    return A00 * A11 - A01 * A10;
  } /* End of 'MatrDeterm3x3' function */

  __device__ float operator!(void) const {
    return M[0][0] * MatrDeterm2x2(M[1][1], M[1][2], M[2][1], M[2][2]) -
           M[0][1] * MatrDeterm2x2(M[1][0], M[1][2], M[2][0], M[2][2]) +
           M[0][2] * MatrDeterm2x2(M[1][0], M[1][1], M[2][0], M[2][1]);
  } /* End of 'operator!' function */

  __device__ d_matr3 Inverse(void) const {
    const float det = !*this;
    if (det != 0)
      return d_matr3(MatrDeterm2x2(M[1][1], M[1][2], M[2][1], M[2][2]) / det,
                     -MatrDeterm2x2(M[0][1], M[0][2], M[2][1], M[2][2]) / det,
                     MatrDeterm2x2(M[0][1], M[0][2], M[1][1], M[1][2]) / det,
                     -MatrDeterm2x2(M[1][0], M[1][2], M[2][0], M[2][2]) / det,
                     MatrDeterm2x2(M[0][0], M[0][2], M[2][0], M[2][2]) / det,
                     -MatrDeterm2x2(M[0][0], M[0][2], M[1][0], M[1][2]) / det,
                     MatrDeterm2x2(M[1][0], M[1][1], M[2][0], M[2][1]) / det,
                     -MatrDeterm2x2(M[0][0], M[0][1], M[2][0], M[2][1]) / det,
                     MatrDeterm2x2(M[0][0], M[0][1], M[1][0], M[1][1]) / det);
    return d_matr3(1, 0, 0, 0, 1, 0, 0, 0, 1);
  } /* End of 'Inverse' function */

  __device__ d_vec TransformVector(const d_vec V) {
    return d_vec(M[0][0] * V.x + M[0][1] * V.y + M[0][2] * V.z,
                 M[1][0] * V.x + M[1][1] * V.y + M[1][2] * V.z,
                 M[2][0] * V.x + M[2][1] * V.y + M[2][2] * V.z);
  } /* End of 'TransformVector' function */
};  /* End of 'd_matr' class */

static __device__ d_vec V_cast(const DWORD D) {
  return d_vec((D >> 16 & 0xFF) / 255.0, (D >> 8 & 0xFF) / 255.0,
               (D >> 0 & 0xFF) / 255.0);
} /* End of 'V_cast' function */

static __device__ float F_cast(const DWORD D) {
  return float(D);
} /* End of 'F_cast' function */

/* Device pair class */
template <class Type1, class Type2> class d_pair {
public:
  Type1 first;  // The first variable
  Type2 second; // The second variable

  __device__ d_pair<Type1, Type2>(const Type1 &a, const Type2 &b) {
    first = a;
    second = b;
  } /* End of 'd_pair' function */
};  /* End of 'd_pair' class */

template <class Type> static __host__ Type h_min(const Type a, const Type b) {
  return a >= b ? b : a;
} /* End of 'h_min' function */

static __device__ float d_min(const float a, const float b) {
  return a >= b ? b : a;
} /* End of 'd_min' function */

static __device__ float d_max(const float a, const float b) {
  return a <= b ? b : a;
} /* End of 'd_max' function */

static __device__ d_vec RGB2HSV(const d_vec C) {
  float maxc = max(max(C.x, C.y), C.z), minc = min(min(C.x, C.y), C.z),
        delta = maxc - minc, S = 0, H, V;
  if (maxc > 0)
    S = delta / maxc;
  V = maxc;
  if (S == 0)
    H = 0;
  else {
    float rc = (maxc - C.x) / delta, gc = (maxc - C.y) / delta,
          bc = (maxc - C.z) / delta;
    if (C.x == maxc)
      H = bc - gc;
    else if (C.y == maxc)
      H = 2 + rc - bc;
    else
      H = 4 + gc - rc;
    H *= 60.0;
  }
  return d_vec(H, S, V);
} /* End of 'RGB2HSV' function */

static __device__ d_vec RGB2HSV(const DWORD D) {
  d_vec C = V_cast(D);
  return RGB2HSV(C);
} /* End of 'RGB2HSV' function */

static __device__ d_vec RGB2XYZ(const d_vec D) {
  d_matr3 M(0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192,
            0.9505);
  return M.TransformVector(D);
} /* End of 'RGB2XYZ' function */

static __device__ float f(const float x) {
  const float c = 6.0 / 29.0;
  if (x > pow(c, 3.0))
    return pow(x, 1.0 / 3.0);
  else
    return x / (3 * c * c) + (4.0 / 29.0);
} /* End of 'f' function */

static __device__ d_vec RGB2LAB(const d_vec D, const d_vec white_point) {
  float L = 0, a = 0, b = 0;
  d_vec basic_c = RGB2XYZ(D);
  L = 116.0 * f(basic_c.y / white_point.y) - 16.0;
  a = 500.0 * (f(basic_c.x / white_point.x) - f(basic_c.y / white_point.y));
  b = 200.0 * (f(basic_c.y / white_point.y) - f(basic_c.z / white_point.z));
  return d_vec(L, a, b);
} /* End of 'RGB2LAB' function */

static __device__ d_vec RGB2LAB(const DWORD D, const d_vec white_point) {
  d_vec C = V_cast(D);
  return RGB2LAB(C, white_point);
} /* End of 'RGB2LAB' function */


#endif /* __cuda_test_cuh_ */
