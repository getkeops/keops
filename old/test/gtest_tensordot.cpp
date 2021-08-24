#include <iostream>
#include <assert.h>
#include <algorithm>

#include "gtest/gtest.h"

#include "keops_includes.h"

using namespace keops;

namespace {


TEST(tensordot, one){

  auto x = Vi(0,2*2*2); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,2*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(2,2,2), Ind(2,2), Ind(2), Ind(0));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FA[8] = {4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  __TYPE__ out_keops[8];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FA, FB);

  double out_loop[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  size_t q =0 ;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 2; k++, q++)
        for (size_t l = 0; l < 2; l++) {
          // size_t kda = 4 * i + 2 * j + l;
          // size_t kdb = l * 2 + k;
          // size_t I = 4 * i + 2 * j + k;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          //out_loop[4 * i + 2 * j + k] += FA[4 * i + 2 * j + l] * FB[l * 2 + k];
          out_loop[q] += FA[4 * i + 2 * j + l] * FB[l * 2 + k];
        }
    }
#else
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 2; k++)
        for (size_t l = 0; l < 2; l++) {
          // size_t kda = 4 * i + 2 * j + l;
          // size_t kdb = l * 2 + k;
          // size_t I = 4 * i + 2 * j + k;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[4 * k + 2 * j + i] += FA[4 * l + 2 * j + i] * FB[k * 2 + l];
        }
    }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<8; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


/*
import numpy
a = np.array([4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3]).reshape(2,2,2)
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(2,2)
np.tensordot(a, b, axes=([2],[0])).flatten() # array([14.26 , 11.868, 18.43 , 15.37 , 19.65 , 16.442, 14.13 , 12.026])

import numpy
a = np.array([4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3]).reshape(2,2,2, order='F')
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(2,2, order='F')
np.tensordot(a, b, axes=([2],[0])).flatten(order='F') # array([15.16 , 14.88 , 19.12 , 10.66 , 15.75 , 15.542, 19.914, 11.336])
*/

TEST(tensordot, two){

  auto x = Vi(0,2*2*2); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,2*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(2,2,2), Ind(2,2), Ind(1,2), Ind(0,1));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FA[8] = {4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  __TYPE__ out_keops[2];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FA, FB);

  __TYPE__ out_loop[2] = {0, 0};

#if C_CONTIGUOUS
  size_t q = 0;
  for (size_t i = 0; i < 2; i++) {
    size_t qq = 0;
    for (size_t k = 0; k < 2; k++)
      for (size_t l = 0; l < 2; l++, q++, qq++) {
        // out_loop[i] += FA[4 * i + 2 * k + l] * FB[k * 2 + l];
        out_loop[i] += FA[q] * FB[qq];
      }
  }
#else
  for (size_t i = 0; i < 2; i++)
    for (size_t k = 0; k < 2; k++)
      for (size_t l = 0; l < 2; l++)
        out_loop[i] += FA[4 * l + 2 * k + i] * FB[l * 2 + k];
#endif
  __TYPE__ s2d = 0;
  for(int i=0; i<2; i++) {
    std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-06);
}


/*
import numpy
a = np.array([4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3]).reshape(2,2,2)
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(2,2)
np.tensordot(a, b, axes=([1,2],[0,1])).flatten() # array([29.87 , 32.456])

import numpy
a = np.array([4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3]).reshape(2,2,2, order='F')
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(2,2, order='F')
np.tensordot(a, b, axes=([1,2],[0,1])).flatten(order='F') # array([35.464, 26.096])
*/


TEST(tensordot, three){

  auto x = Vi(0,5*4*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4*3*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(5, 4, 3), Ind(4, 3, 2), Ind(1, 2), Ind(0, 1));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

  __TYPE__ out_keops[10];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
          // size_t kda = 12 * i + 3 * k + l;
          // size_t kdb = 6 * k + 2 * l + j;
          // size_t I = 2 * i + j;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[2 * i + j] += FAA[12 * i + 3 * k + l] * FBB[6 * k + 2 * l + j];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
          // size_t kda = 20 * l + 5 * k + i;
          // size_t kdb = 12 * j + 4 * l + k;
          // size_t I = 5 * j + i;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[5 * j + i] += FAA[20 * l + 5 * k + i] * FBB[12 * j + 4 * l + k];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8.]).reshape(5, 4, 3)
b = np.array([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7.]).reshape(4, 3, 2)
np.tensordot(a, b, axes=([1, 2], [0, 1])).flatten() # array([357., 499., 226., 270., 160., 328., 256., 386., 274., 401.])

import numpy
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8.]).reshape(5, 4, 3, order='F')
b = np.array([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7.]).reshape(4, 3, 2, order='F')
np.tensordot(a, b, axes=([1, 2], [0, 1])).flatten(order='F') #array([412., 315., 290., 259., 311., 389., 306., 256., 236., 288.])
*/


TEST(tensordot, four){

  auto x = Vi(0,5*4*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,5*4*3); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(5, 4, 3), Ind(5, 4, 3), Ind(0, 1, 2), Ind(0, 1, 2));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};

  __TYPE__ out_keops;
  EvalRed<CpuConv>(Sum_f,1, 1, &out_keops, FAA, FAA);

  __TYPE__ out_loop = 0;
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 4; j++)
      for (size_t k = 0; k < 3; k++)
          out_loop += FAA[12 * i + 3 * j + k] * FAA[12 * i + 3 * j + k];

  __TYPE__ s2d = abs(out_keops - out_loop);

  EXPECT_LE(s2d,5e-6);
}

/*
import numpy
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8.])
(a * a).sum() # 1968
*/

TEST(tensordot, five){

  auto x = Vi(0,4*5*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,3*4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(4, 5, 3), Ind(3, 4, 2), Ind(0, 2), Ind(1, 0));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

  __TYPE__ out_keops[10];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
          // size_t kda = 15 * k + 3 * i + l;
          // size_t kdb = 8 * l + 2 * k + j;
          // size_t I = 2 * i + j;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[2 * i + j] += FAA[15 * k + 3 * i + l] * FBB[8 * l + 2 * k + j];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
          // size_t kda = 20 * k + 4 * i + k;
          // size_t kdb = 12 * j + 3 * k + l;
          // size_t I = 5 * j + i;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[5 * j + i] += FAA[20 * l + 4 * i + k] * FBB[12 * j + 3 * k + l];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

 /*
import numpy as np
a= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]],[[3, 9, 8],[8, 3, 7],[2, 3, 1],[9, 5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3)
b = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2)
np.tensordot(a,b,axes=([0,2],[1,0])).flatten() # array([318, 405, 267, 392, 222, 389, 269, 391, 174, 277])

import numpy as np
a= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]],[[3, 9, 8],[8, 3, 7],[2, 3, 1],[9, 5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3, order='F')
b = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2, order='F')
np.tensordot(a,b,axes=([0,2],[1,0])).flatten(order='F') # array([335, 354, 289, 252, 337, 348, 331, 293, 239, 327])
*/


TEST(tensordot, six){

  auto x = Vi(0,4*5*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,3*4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(4, 5, 3), Ind(3, 4, 2), Ind(2, 0), Ind(0, 1));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

  __TYPE__ out_keops[10];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 4; l++) {
          // size_t kda = 15 * l + 3 * i + k;
          // size_t kdb = 8 * k + 2 * l + j;
          // size_t I = 2 * i + j;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[2 * i + j] += FAA[15 * l + 3 * i + k] * FBB[8 * k + 2 * l + j];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 4; l++) {
          // size_t kda = 20 * k + 4 * i + l;
          // size_t kdb = 12 * j + 3 * l + k;
          // size_t I = 5 * j + i;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[5 * j + i] += FAA[20 * k + 4 * i + l] * FBB[12 * j + 3 * l + k];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy as np
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8]).reshape(4, 5, 3)
b = np.array([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7]).reshape(3, 4, 2)
np.tensordot(a,b,axes=([2,0],[0,1])).flatten() # array([318, 405, 267, 392, 222, 389, 269, 391, 174, 277])

import numpy as np
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8]).reshape(4, 5, 3, order='F')
b = np.array([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7]).reshape(3, 4, 2, order='F')
np.tensordot(a,b,axes=([2,0],[0,1])).flatten(order='F') # array([335, 354, 289, 252, 337, 348, 331, 293, 239, 327])
*/


TEST(tensordot, seven){

  auto x = Vi(0,4*5*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,3*4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(4, 5, 3), Ind(3, 4, 2), Ind(0), Ind(1));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

  __TYPE__ out_keops[90];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[90] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 2; l++) 
          for (size_t m = 0; m < 4; m++) {
            // size_t kda = 15 * l + 3 * i + k;
            // size_t kdb = 8 * k + 2 * l + j;
            // size_t I = 2 * i + j;
            // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
            out_loop[18 * i + 6 * j + 2 * k + l] += FAA[15 * m + 3 * i + j] * FBB[8 * k + 2 * m + l];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 2; l++)
          for (size_t m = 0; m < 4; m++) {
            // size_t I = 45 * l + 15 * k + 5 * j + i;
            // size_t kda = 20 * j + 4 * i + m;
            // size_t kdb = 12 * l + 3 * m + k;
            // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
            out_loop[45 * l + 15 * k + 5 * j + i] += FAA[20 * j + 4 * i + m] * FBB[12 * l + 3 * m + k];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<90; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy as np
a= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]], [[3, 9, 8],[8, 3, 7],[2, 3, 1],[9, 5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3)
b = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2)

np.tensordot(a,b,axes=([0],[1])).flatten()
 
# array([115, 157, 128, 202, 113, 149,  74,  98, 105, 133,  82,  85,  94,
#        120, 121, 167,  98, 115,  46,  67,  85, 105,  63,  77, 107, 163,
#        113, 176, 104, 117, 108, 182, 109, 195, 108, 149,  82, 135, 106,
#        155,  92, 109, 135,  81,  72, 153,  72, 108, 115,  97,  63, 140,
#         68, 101,  87, 121,  77, 156,  78, 133,  78, 140, 101, 151,  90,
#        107, 110,  93,  91, 159,  81, 119,  52,  94,  39,  74,  45,  44,
#         66, 103,  64, 107,  62,  73,  35,  65,  78,  97,  58,  76])

import numpy as np
aa= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]], [[3, 9, 8],[8, 3, 7],[2, 3, 1],[9,5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3, order='F')
bb = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2, order='F')

np.tensordot(aa,bb,axes=([0],[1])).flatten(order='F')

# array([172, 153,  97,  97, 117, 132, 145,  50,  87, 171, 107, 148, 152,
#         74,  97, 173, 113,  68,  76,  57,  96,  91,  62,  59, 157,  93,
#        129, 141,  77,  54, 142, 109,  75,  67,  57,  60,  73,  58,  67,
#        139,  67, 110, 130,  96,  63, 146, 122,  77,  84,  99,  81, 123,
#         32,  79, 163,  89, 144, 130,  91,  78, 151, 144,  99,  78,  87,
#        126, 102,  71,  75, 128,  81,  99, 141,  58,  91, 147, 149, 106,
#         81,  96, 108, 105,  67,  86, 137,  76, 107, 145,  80, 100])
*/


TEST(tensordot, height){

  auto x = Vi(0,4); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(4), Ind(4), Ind(), Ind());
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FA[4] = {4.4, 5.4, 6.2, 6.5};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  __TYPE__ out_keops[16];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FA, FB);

  double out_loop[16];
#if C_CONTIGUOUS
  size_t q =0 ;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++, q++) {
      out_loop[q] = FA[i] * FB[j];
      }
    }
#else
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      out_loop[4 * j + i] = FA[i] * FB[j];
      }
    }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<16; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy as np
a = np.array([4.4, 5.4, 6.2, 6.5])
b = np.array([1.4, 1.2, 1.5, 1.22])
(a[:,None] @ b[None,:]).flatten()

import numpy as np
a = np.array([4.4, 5.4, 6.2, 6.5])
b = np.array([1.4, 1.2, 1.5, 1.22])
(a[:,None] @ b[None,:]).flatten(order='F')
*/


TEST(tensordot, nine){

  __TYPE__ FA[4] = {4.4, 5.4, 6.2, 6.5};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};
  __TYPE__ XI[16] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8};


  auto x = Vi(0,4); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto xi = Vj(2,16); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = Grad(TensorDot(x,y, Ind(4), Ind(4), Ind(), Ind()),x,xi);
  __TYPE__ out_keops[4];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB, XI);

  auto f_legacy = Grad(TensorProd(x,y),x,xi);
  __TYPE__ out_legacy[4];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB, XI);

  __TYPE__ s2d = 0;
  for(int i=0; i<4; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import torch
a = torch.tensor([4.4, 5.4, 6.2, 6.5], requires_grad=True)
b = torch.tensor([1.4, 1.2, 1.5, 1.22], requires_grad=True)
c = a[:,None] @ b[None,:]
xi = torch.tensor([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8.]).reshape(4,4)
torch.autograd.grad(c, a, xi) # (tensor([27.1800, 27.4200, 27.2800, 32.4600]),

# Torch does not support order='F' options. We have to use .transpose() method to mimic its behaviour
import torch
a = torch.tensor([4.4, 5.4, 6.2, 6.5], requires_grad=True)
b = torch.tensor([1.4, 1.2, 1.5, 1.22], requires_grad=True)
c = a[:,None] @ b[None,:]
xi = torch.tensor([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8.]).reshape(4,4).transpose(0,1)
torch.autograd.grad(c, a, xi) # (tensor([30.9200, 34.5800, 15.5400, 35.5600]),)
*/


TEST(tensordot, ten){

  __TYPE__ FA[4] = {4.4, 5.4, 6.2, 6.5};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};
  __TYPE__ XI[16]  = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8};

  auto x = Vi(0,4); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto xi = Vi(2,16); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = Grad(TensorDot(x,y, Ind(4), Ind(4), Ind(), Ind()),y,xi);
  __TYPE__ out_keops[4];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB, XI);

  auto f_legacy = Grad(TensorProd(x,y),y,xi);
  __TYPE__ out_legacy[4];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB, XI);

  __TYPE__ s2d = 0;
  for(int i=0; i<4; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import torch
a = torch.tensor([4.4, 5.4, 6.2, 6.5], requires_grad=True)
b = torch.tensor([1.4, 1.2, 1.5, 1.22], requires_grad=True)
c = a[:,None] @ b[None,:]
xi = torch.tensor([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8.]).reshape(4,4)
torch.autograd.grad(c, b, xi) #(tensor([124.9000, 152.7000,  72.1000, 148.8000]),)

# Torch does not support order='F' options. We have to use .transpose() method to mimic its behaviour
import torch
a = torch.tensor([4.4, 5.4, 6.2, 6.5], requires_grad=True)
b = torch.tensor([1.4, 1.2, 1.5, 1.22], requires_grad=True)
c = a[:,None] @ b[None,:]
xi = torch.tensor([6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8.]).reshape(4,4).transpose(0,1)
torch.autograd.grad(c, b, xi) # (tensor([118.9000, 111.8000, 112.4000, 148.4000]),)
*/


TEST(tensordot, eleven){

  __TYPE__ FA[16] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  auto x = Vi(0,16); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = TensorDot(x,y, Ind(4,4), Ind(4), Ind(1), Ind(0));
  __TYPE__ out_keops[4];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB);

  auto f_legacy = MatVecMult(x,y);
  __TYPE__ out_legacy[4];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB);

  __TYPE__ s2d = 0;
  for(int i=0; i<4; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


TEST(tensordot, twelve){

  __TYPE__ FA[16] = {6, 44, 20, 9, 99, 5, 1, 6, 7, 8, 2, 4, 1.1, 55.9, 7, 8};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  auto x = Vi(0,16); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = TensorDot(y,x, Ind(4), Ind(4,4), Ind(0), Ind(0));
  __TYPE__ out_keops[4];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB);

  auto f_legacy = VecMatMult(y,x);
  __TYPE__ out_legacy[4];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB);

  __TYPE__ s2d = 0;
  for(int i=0; i<4; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy
a = np.array([6, 44, 20, 9, 99, 5, 1, 6, 7, 8, 2, 4, 1.1, 55.9, 7, 8]).reshape(4,4)
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(1,4)
b @ a # array([[139.042, 147.798,  40.74 ,  35.56 ]])

import numpy
a = np.array([6, 44, 20, 9, 99, 5, 1, 6, 7, 8, 2, 4, 1.1, 55.9, 7, 8]).reshape(4,4, order='F')
b = np.array([1.4, 1.2, 1.5, 1.22]).reshape(1,4, order='F')
b @ a # array([[102.18, 153.42,  27.28,  88.88]])
*/


TEST(tensordot, thirteen){

  __TYPE__ FA[16] = {7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7.43, 8.7, 2.21, 4.98, 1.2, 9.32, 7.76, 8.33};
  __TYPE__ FB[4] = {2.4, 1.2, 1.5, 1.22};
  __TYPE__ XI[4] = { 4.4, 2.4, 6.65, 5.5};

  auto x = Vi(0,16); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto xi = Vj(2,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = Grad(TensorDot(x,y, Ind(4,4), Ind(4), Ind(1), Ind(0)),x,xi);
  //auto f = TensorDot(x,y, Ind(4,4), Ind(4), Ind(1), Ind(0));
  __TYPE__ out_keops[16];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB, XI);

  auto f_legacy = Grad(MatVecMult(x,y),x,xi);
  __TYPE__ out_legacy[16];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB, XI);

  __TYPE__ s2d = 0;
#if C_CONTIGUOUS
  for(int i=0; i<16; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
#else
    for(int i=0; i<16; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
#endif
  EXPECT_LE(s2d,5e-6);
}

/*
import torch
a = torch.tensor([7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7.43, 8.7, 2.21, 4.98, 1.2, 9.32, 7.76, 8.33], requires_grad=True).reshape(4,4)
b = torch.tensor([2.4, 1.2, 1.5, 1.22], requires_grad=True).reshape(4,1)
c = a @ b
xi = torch.tensor([ 4.4, 2.4, 6.65, 5.5]).reshape(4,1)
torch.autograd.grad(c, a, xi)[0].view(-1) # tensor([10.5600,  5.2800,  6.6000,  5.3680,  5.7600,  2.8800,  3.6000,  2.9280, 15.9600,  7.9800,  9.9750,  8.1130, 13.2000,  6.6000,  8.2500,  6.7100])


# Torch does not support order='F' options. We have to use .transpose() method to mimic its behaviour
import torch
a = torch.tensor([7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7.43, 8.7, 2.21, 4.98, 1.2, 9.32, 7.76, 8.33], requires_grad=True).reshape(4,4).transpose(0,1)
b = torch.tensor([2.4, 1.2, 1.5, 1.22], requires_grad=True).reshape(4,1)
c = a @ b
xi = torch.tensor([ 4.4, 2.4, 6.65, 5.5]).reshape(1,4)
torch.autograd.grad(c.transpose(0,1), a, xi)[0].view(-1) # tensor([116.5350,  97.1100,  85.3000, 116.5500])
*/


TEST(tensordot, fourteen){

  __TYPE__ FA[16] = {7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7, 8, 2, 4, 1, 9, 7, 8};
  __TYPE__ FB[4] = {2.4, 1.2, 1.5, 1.22};
  __TYPE__ XI[4] = { 4.4, 2.4, 6.65, 5.5};

  auto x = Vi(0,16); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto xi = Vi(2,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = Grad(TensorDot(x,y, Ind(4,4), Ind(4), Ind(1), Ind(0)),y,xi);
  //auto f = TensorDot(x,y, Ind(4,4), Ind(4), Ind(1), Ind(0));
  __TYPE__ out_keops[4];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB, XI);

  auto f_legacy = Grad(MatVecMult(x,y),y,xi);
  __TYPE__ out_legacy[4];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB, XI);

  __TYPE__ s2d = 0;
  for(int i=0; i<4; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import torch
a = torch.tensor([7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7, 8, 2, 4, 1, 9, 7, 8], requires_grad=True).reshape(4,4)
b = torch.tensor([2.4, 1.2, 1.5, 1.22], requires_grad=True).reshape(4,1)
c = a @ b
xi = torch.tensor([ 4.4, 2.4, 6.65, 5.5]).reshape(4,1)
torch.autograd.grad(c, b, xi)[0].view(-1) #tensor([108.2500, 135.3160,  67.4240, 128.1200])

# Torch does not support order='F' options. We have to use .transpose() method to mimic its behaviour
import torch
a = torch.tensor([7.7, 4.5, 2.7, 9.8, 9.3, 5.34, 1.56, 6, 7, 8, 2, 4, 1, 9, 7, 8], requires_grad=True).reshape(4,4).transpose(0,1)
b = torch.tensor([2.4, 1.2, 1.5, 1.22], requires_grad=True).reshape(4,1)
c = a @ b
xi = torch.tensor([ 4.4, 2.4, 6.65, 5.5]).reshape(1,4)
torch.autograd.grad(c.transpose(0,1), b, xi)[0].view(-1) # tensor([116.5350,  97.1100,  85.3000, 116.5500])
*/

TEST(tensordot, fifteen){

  __TYPE__ FA[4] = {4.4, 5.4, 6.2, 6.5};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  auto x = Vi(0,4); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

  auto f = TensorDot(x,y, Ind(4), Ind(4), Ind(), Ind());
  __TYPE__ out_keops[16];
  EvalRed<CpuConv>(Sum_Reduction(f,0),1, 1, out_keops, FA, FB);

  auto f_legacy = TensorProd(x,y);
  __TYPE__ out_legacy[16];
  EvalRed<CpuConv>(Sum_Reduction(f_legacy,0),1, 1, out_legacy, FA, FB);

  __TYPE__ s2d = 0;
  for(int i=0; i<16; i++) {
    // std::cout << out_keops[i] << "      " << out_legacy[i] << std::endl;
    s2d += abs(out_legacy[i] - out_keops[i]);
  }
  EXPECT_LE(s2d,5e-6);
}



TEST(tensordot, sixteen){

  auto x = Vi(0,5*4*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(5, 4, 3), Ind(4, 2), Ind(1), Ind(0), Ind(0, 2, 1));
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[8] = {6, 4, 2, 9, 9, 5, 1, 6};

  __TYPE__ out_keops[30];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[30] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 2; k++)
        for (size_t l = 0; l < 4; l++) {
          // size_t kda = 12 * i + 3 * l + j;
          // size_t kdb =  2 * l + k;
          // size_t I = 6 * i + 3 * k + j;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[6 * i + 3 * k + j] += FAA[12 * i + 3 * l + j] * FBB[ 2 * l + k];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 2; k++)
        for (size_t l = 0; l < 4; l++) {
          // size_t kda = 20 * l + 5 * k + i;
          // size_t kdb = 12 * j + 4 * l + k;
          // size_t I = 5 * j + i;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[10 * j + 5 * k + i] += FAA[20 * j + 5 * l + i] * FBB[4 * k + l];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<30; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

/*
import numpy as np
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8]).reshape(5, 4, 3)
b = np.array([6, 4, 2, 9, 9, 5, 1, 6]).reshape(4, 2)
np.tensordot(a,b,axes=([1],[0])).swapaxes(2,1).flatten()
# array([106, 156, 121, 103, 183, 135,  34, 111, 108,  93,  88, 102,  89, 67,  34, 120, 111,  57,  61,  92,  78, 148, 108, 142, 137,  96, 109, 136,  73, 118])


import numpy as np
a = np.array([7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8]).reshape(5, 4, 3, order='F')
b = np.array([6, 4, 2, 9, 9, 5, 1, 6]).reshape(4, 2, order='F')
np.tensordot(a,b,axes=([1],[0])).swapaxes(2,1).flatten(order='F')
# array([109, 119, 123,  62, 135, 113, 136, 147,  79, 129, 157,  65, 119, 116,  98, 164,  73,  97, 106,  79, 135, 121,  40,  75, 116, 123, 125,  53,  81,  91])
*/




/*
TEST(tensordot, seventeen){

  auto x = Vi(0, 2*3*4  ); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  // auto xi = Vj(2,3);
  //auto Sum_f = Sum_Reduction( Grad(TensorDot(x, y, Ind(2,3,4), Ind(4,2), Ind(2,0), Ind(0,1), Ind(0)), x, xi),0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")
  auto xi = Vi(2,3);
  auto Sum_f = Sum_Reduction( Grad(TensorDot(x, y, Ind(2,3,4), Ind(4,2), Ind(2,0), Ind(0,1), Ind(0)), y, xi),0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[24] = {6, 4, 2, 9, 9, 5, 1, 6,6, 4, 2, 9, 9, 5, 1, 6,6, 4, 2, 9, 9, 5, 1, 6};
  __TYPE__ FBB[8] = {6, 4, 2, 9, 9, 5, 1, 6};
  __TYPE__ XI[3] = {6, 4, 2};
  __TYPE__ out_keops[8];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB, XI);


  __TYPE__ s2d = 0;
  for(int i=0; i<8; i++) {
    std::cout << out_keops[i] << std::endl;
    s2d += abs(out_keops[i] );
  }
  EXPECT_LE(s2d,5e-6);
}
*/

TEST(tensordot, seventeen){

  auto x = Vi(0,4*5*3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,3*4*2); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  //auto f = TensorDot(x,y, Ind(4, 5, 3), Ind(3, 4, 2), Ind(0, 2), Ind(1, 0), Ind(0,1));
  auto f = TensorDot(x,y, Ind(4, 5, 3), Ind(3, 4, 2), Ind(0, 2), Ind(1, 0), Ind(1,0));

  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
  __TYPE__ FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

  __TYPE__ out_keops[10];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FAA, FBB);

  __TYPE__ out_loop[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#if C_CONTIGUOUS
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
            // size_t kda = 15 * k + 3 * i + l;
            // size_t kdb = 8 * l + 2 * k + j;
            // size_t I = 2 * i + j;
            // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
            out_loop[5 * j + i] += FAA[15 * k + 3 * i + l] * FBB[8 * l + 2 * k + j];
        }
#else
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
            // size_t I = 5 * j + i;
            // size_t kda = 20 * l + 4 * i + k;
            // size_t kdb = 12 * j + 3 * k + l;
            // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
            out_loop[2 * i + j] += FAA[20 * l + 4 * i + k] * FBB[12 * j + 3 * k + l];
        }
#endif

  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);

}
/*
import numpy as np
a= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]], [[3, 9, 8],[8, 3, 7],[2, 3, 1],[9, 5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3)
b = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2)
print(np.tensordot(a,b,axes=([0,2],[1,0])).swapaxes(1,0).flatten())
# [318 267 222 269 174 405 392 389 391 277]

import numpy as np
aa= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]], [[3, 9, 8],[8, 3, 7],[2, 3, 1],[9,5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3, order='F')
bb = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2, order='F')
print(np.tensordot(aa,bb,axes=([0,2],[1,0])).swapaxes(1,0).flatten(order='F'))
# [335 348 354 331 289 293 252 239 337 327]
*/

} // namespace

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
