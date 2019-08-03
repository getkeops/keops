#include <iostream>
#include <assert.h>
#include <algorithm>

#include <gtest/gtest.h>

#include <keops_includes.h>

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

  __TYPE__ s2d = 0;
  for(int i=0; i<8; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


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

  size_t q = 0;
  for (size_t i = 0; i < 2; i++) {
    size_t qq = 0;
    for (size_t k = 0; k < 2; k++)
      for (size_t l = 0; l < 2; l++, q++, qq++) {
        // out_loop[i] += FA[4 * i + 2 * k + l] * FB[k * 2 + l];
        out_loop[i] += FA[q] * FB[qq];
      }
  }
  __TYPE__ s2d = 0;
  for(int i=0; i<2; i++) {
    //std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-10);
}

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
  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
//            size_t kda = 15 * k + 3 * i + l;
//            size_t kdb = 8 * l + 2 * k + j;
//            size_t I = 2 * i + j;
//            std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;

          out_loop[2 * i + j] += FAA[15 * k + 3 * i + l] * FBB[8 * l + 2 * k + j];
        }


  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    //std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}

 /*
a= np.array([[[7, 9, 9], [5, 8, 3], [6, 9, 6], [0, 5, 7]], [[3, 4, 3], [5, 3, 3],[0, 9, 9],[6, 0, 3]],[[3, 7, 0],[8, 6, 0],[6, 1, 3],[1, 4, 7]],[[3, 9, 8],[8, 3, 7],[2, 3, 1],[9, 5, 7]],[[7, 5, 9],[7, 0, 1],[9, 7, 5],[0, 3, 8]]]).flatten().reshape(4, 5, 3)
b = np.array([[[6, 4],[2, 9],[9, 5]],[[1, 6],[7, 8],[2, 4]],[[1, 9],[7, 8],[5, 4]],[[3, 2],[3, 8],[5, 7]]]).flatten().reshape(3,4,2)
np.tensordot(a,b,axes=([0,2],[1,0])).flatten()
# array([318, 405, 267, 392, 222, 389, 269, 391, 174, 277])
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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 4; l++) {
//            size_t kda = 15 * l + 3 * i + k;
//            size_t kdb = 8 * k + 2 * l + j;
//            size_t I = 2 * i + j;
//            std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;

          out_loop[2 * i + j] += FAA[15 * l + 3 * i + k] * FBB[8 * k + 2 * l + j];
        }


  __TYPE__ s2d = 0;
  for(int i=0; i<10; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        for (size_t l = 0; l < 2; l++) 
          for (size_t m = 0; m < 4; m++) {
//            size_t kda = 15 * l + 3 * i + k;
//            size_t kdb = 8 * k + 2 * l + j;
//            size_t I = 2 * i + j;
//            std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << I << " " << kda << " " << kdb  << std::endl;
          out_loop[ 18 * i + 6 * j + 2 * k + l] += FAA[15 * m + 3 * i + j] * FBB[8 * k + 2 * m + l];
        }


  __TYPE__ s2d = 0;
  for(int i=0; i<90; i++) {
    //std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}
/*

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

*/ 



TEST(tensordot, height){

  auto x = Vi(0,4); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
  auto f = TensorDot(x,y, Ind(4), Ind(4), Ind(), Ind());
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

  __TYPE__ FA[8] = {4.4, 5.4, 6.2, 6.5};
  __TYPE__ FB[4] = {1.4, 1.2, 1.5, 1.22};

  __TYPE__ out_keops[16];
  EvalRed<CpuConv>(Sum_f,1, 1, out_keops, FA, FB);

  double out_loop[16];
  size_t q =0 ;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++, q++) {
      out_loop[q] = FA[i] * FB[j];
      }
    }

  __TYPE__ s2d = 0;
  for(int i=0; i<16; i++) {
    // std::cout << out_keops[i] << "      " << out_loop[i] << std::endl;
    s2d += abs(out_keops[i] - out_loop[i]);
  }
  EXPECT_LE(s2d,5e-6);
}


} // namespace

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
