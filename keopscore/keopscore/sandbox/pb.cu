
#define C_CONTIGUOUS 1
#define USE_HALF 0

extern "C" __global__ void
GpuConv1DOnDevice_ranges(signed long int nx, signed long int ny, int nbatchdims,
                         signed long int *offsets_d, signed long int *lookup_d,
                         signed long int *slices_x, signed long int *ranges_y,
                         float *out, float **arg_1) {

  signed long int offsets[4];
  signed long int *indices_i = offsets;
  signed long int *indices_j = offsets + 1;
  signed long int *indices_p = offsets + 1 + 2;

  if (nbatchdims > 0)
    for (int k = 0; k < 4; k++)
      offsets[k] = offsets_d[4 * blockIdx.x + k];
  // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
  signed long int range_id = (lookup_d)[3 * blockIdx.x];
  signed long int start_x = (lookup_d)[3 * blockIdx.x + 1];
  signed long int end_x = (lookup_d)[3 * blockIdx.x + 2];

  // The "slices_x" vector encodes a set of cutting points in
  // the "ranges_y" array of ranges.
  // As discussed in the Genred docstring, the first "0" is implicit:
  signed long int start_slice = range_id < 1 ? 0 : slices_x[range_id - 1];
  signed long int end_slice = slices_x[range_id];

  // get the index of the current thread
  signed long int i = start_x + threadIdx.x;

  // declare shared mem
  extern __shared__ float yj[];

  // load parameter(s)
  float param_loc[1];

  if (nbatchdims == 0) {
    {
      signed long int a = 0;

      for (signed long int v = 0; v < 1; v++) {
        param_loc[a] = arg_1[0][0 * 1 + v];
        a++;
      }
    }

  } else {
    {
      signed long int a = 0;

      for (signed long int v = 0; v < 1; v++) {
        param_loc[a] = arg_1[0][(0 + indices_p[0]) * 1 + v];
        a++;
      }
    }
  }
  float fout[1];
  float xi[4];
  float acc[1];
  float tmp[1];

  if (i < end_x) {
    for (int k_22 = 0; k_22 < 1; k_22 += (1)) {
      acc[k_22] = (0.0f);
    }
    // acc = 0

    if (nbatchdims == 0) {
      {
        signed long int a = 0;

        for (signed long int v = 0; v < 3; v++) {
          xi[a] = arg_1[2][i * 3 + v];
          a++;
        }
      }
      // load xi variables from global memory to local thread memory
    } else {
      {
        signed long int a = 0;

        for (signed long int v = 0; v < 3; v++) {
          xi[a] = arg_1[2][(threadIdx.x + indices_i[0]) * 3 + v];
          a++;
        }
      }
      // Possibly, with offsets as we support broadcasting over batch dimensions
    }
  }

  signed long int start_y = ranges_y[2 * start_slice], end_y = 0;
  for (signed long int index = start_slice; index < end_slice; index++) {
    if ((index + 1 >= end_slice) ||
        (ranges_y[2 * index + 2] != ranges_y[2 * index + 1])) {
      // start_y = ranges_y[2*index] ;
      end_y = ranges_y[2 * index + 1];

      for (signed long int jstart = start_y, tile = 0; jstart < end_y;
           jstart += blockDim.x, tile++) {
        // get the current column
        signed long int j = jstart + threadIdx.x;

        if (j < end_y) // we load yj from device global memory only if j<end_y
          if (nbatchdims == 0) {
            {
              signed long int a = 0;

              for (signed long int v = 0; v < 3; v++) {
                (yj + threadIdx.x * 3)[a] = arg_1[1][j * 3 + v];
                a++;
              }

              for (signed long int v = 0; v < 1; v++) {
                (yj + threadIdx.x * 3)[a] = arg_1[4][j * 1 + v];
                a++;
              }
            }
            // load yj variables from global memory to shared memory
          } else {
            {
              signed long int a = 0;

              for (signed long int v = 0; v < 3; v++) {
                (yj + threadIdx.x * 3)[a] =
                    arg_1[1][((j - start_y) + indices_j[0]) * 3 + v];
                a++;
              }

              for (signed long int v = 0; v < 1; v++) {
                (yj + threadIdx.x * 3)[a] =
                    arg_1[4][((j - start_y) + indices_j[1]) * 1 + v];
                a++;
              }
            }
            // Possibly, with offsets as we support broadcasting over batch
            // dimensions
          }
        __syncthreads();

        if (i < end_x) {     // we compute x1i only if needed
          float *yjrel = yj; // Loop on the columns of the current block.
          for (int k_23 = 0; k_23 < 1; k_23 += (1)) {
            tmp[k_23] = (0.0f);
          }

          if (nbatchdims == 0) {
            for (signed long int jrel = 0;
                 (jrel < blockDim.x) && (jrel < end_y - jstart);
                 jrel++, yjrel += 3) {

              {
                // Starting code block for
                // Var(4,1,0)*Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).

                float out_exp_2[1];

                {
                  // Starting code block for
                  // Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).

                  float out_minus_2[1];

                  {
                    // Starting code block for
                    // -(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2)).

                    float out_mult_2[1];

                    {
                      // Starting code block for
                      // Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2).

                      float out_sum_2[1];

                      {
                        // Starting code block for
                        // Sum((Var(1,3,0)-Var(2,3,1))**2).

                        float out_square_2[3];

                        {
                          // Starting code block for (Var(1,3,0)-Var(2,3,1))**2.

                          float out_subtract_2[3];

                          {
                            // Starting code block for Var(1,3,0)-Var(2,3,1).

                            for (int k_24 = 0; k_24 < 3; k_24 += (1)) {
                              out_subtract_2[(k_24 * 1)] =
                                  (yjrel + 0)[(k_24 * 1)] -
                                  (xi + 0)[(k_24 * 1)];
                            }

                            // Finished code block for Var(1,3,0)-Var(2,3,1).
                          }

                          for (int k_25 = 0; k_25 < 3; k_25 += (1)) {
                            out_square_2[(k_25 * 1)] =
                                ((out_subtract_2[(k_25 * 1)] *
                                  out_subtract_2[(k_25 * 1)]));
                          }

                          // Finished code block for (Var(1,3,0)-Var(2,3,1))**2.
                        }

                        for (int k_26 = 0; k_26 < 1; k_26 += (1)) {
                          out_sum_2[k_26] = (0.0f);
                        }

                        for (int k_27 = 0; k_27 < 3; k_27 += (1)) {
                          out_sum_2[(k_27 * 0)] += (out_square_2[(k_27 * 1)]);
                        }

                        // Finished code block for
                        // Sum((Var(1,3,0)-Var(2,3,1))**2).
                      }

                      for (int k_28 = 0; k_28 < 1; k_28 += (1)) {
                        out_mult_2[(k_28 * 1)] = (
#ifdef __CUDACC__
                            ((param_loc + 0)[(k_28 * 1)] *
                             out_sum_2[(k_28 * 1)])
#else
                            ((param_loc + 0)[(k_28 * 1)] *
                             out_sum_2[(k_28 * 1)])
#endif
                        );
                      }

                      // Finished code block for
                      // Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2).
                    }

                    for (int k_29 = 0; k_29 < 1; k_29 += (1)) {
                      out_minus_2[(k_29 * 1)] = -out_mult_2[(k_29 * 1)];
                    }

                    // Finished code block for
                    // -(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2)).
                  }

                  for (int k_30 = 0; k_30 < 1; k_30 += (1)) {
                    out_exp_2[(k_30 * 1)] = (
#ifdef __CUDACC__
                        exp(out_minus_2[(k_30 * 1)])
#else
                        exp(out_minus_2[(k_30 * 1)])
#endif
                    );
                  }

                  // Finished code block for
                  // Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).
                }

                for (int k_31 = 0; k_31 < 1; k_31 += (1)) {
                  fout[(k_31 * 1)] = (
#ifdef __CUDACC__
                      ((yjrel + 3)[(k_31 * 1)] * out_exp_2[(k_31 * 1)])
#else
                      ((yjrel + 3)[(k_31 * 1)] * out_exp_2[(k_31 * 1)])
#endif
                  );
                }

                // Finished code block for
                // Var(4,1,0)*Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).
              }

              // Call the function, which outputs results in xi[0:DIMX1]

              for (int k_32 = 0; k_32 < 1; k_32 += (1)) {
                tmp[(k_32 * 1)] += (fout[(k_32 * 1)]);
              }
            }
          } else {
            for (signed long int jrel = 0;
                 (jrel < blockDim.x) && (jrel < end_y - jstart);
                 jrel++, yjrel += 3) {

              {
                // Starting code block for
                // Var(4,1,0)*Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).

                float out_exp_3[1];

                {
                  // Starting code block for
                  // Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).

                  float out_minus_3[1];

                  {
                    // Starting code block for
                    // -(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2)).

                    float out_mult_3[1];

                    {
                      // Starting code block for
                      // Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2).

                      float out_sum_3[1];

                      {
                        // Starting code block for
                        // Sum((Var(1,3,0)-Var(2,3,1))**2).

                        float out_square_3[3];

                        {
                          // Starting code block for (Var(1,3,0)-Var(2,3,1))**2.

                          float out_subtract_3[3];

                          {
                            // Starting code block for Var(1,3,0)-Var(2,3,1).

                            for (int k_33 = 0; k_33 < 3; k_33 += (1)) {
                              out_subtract_3[(k_33 * 1)] =
                                  (yjrel + 0)[(k_33 * 1)] -
                                  (xi + 0)[(k_33 * 1)];
                            }

                            // Finished code block for Var(1,3,0)-Var(2,3,1).
                          }

                          for (int k_34 = 0; k_34 < 3; k_34 += (1)) {
                            out_square_3[(k_34 * 1)] =
                                ((out_subtract_3[(k_34 * 1)] *
                                  out_subtract_3[(k_34 * 1)]));
                          }

                          // Finished code block for (Var(1,3,0)-Var(2,3,1))**2.
                        }

                        for (int k_35 = 0; k_35 < 1; k_35 += (1)) {
                          out_sum_3[k_35] = (0.0f);
                        }

                        for (int k_36 = 0; k_36 < 3; k_36 += (1)) {
                          out_sum_3[(k_36 * 0)] += (out_square_3[(k_36 * 1)]);
                        }

                        // Finished code block for
                        // Sum((Var(1,3,0)-Var(2,3,1))**2).
                      }

                      for (int k_37 = 0; k_37 < 1; k_37 += (1)) {
                        out_mult_3[(k_37 * 1)] = (
#ifdef __CUDACC__
                            ((param_loc + 0)[(k_37 * 1)] *
                             out_sum_3[(k_37 * 1)])
#else
                            ((param_loc + 0)[(k_37 * 1)] *
                             out_sum_3[(k_37 * 1)])
#endif
                        );
                      }

                      // Finished code block for
                      // Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2).
                    }

                    for (int k_38 = 0; k_38 < 1; k_38 += (1)) {
                      out_minus_3[(k_38 * 1)] = -out_mult_3[(k_38 * 1)];
                    }

                    // Finished code block for
                    // -(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2)).
                  }

                  for (int k_39 = 0; k_39 < 1; k_39 += (1)) {
                    out_exp_3[(k_39 * 1)] = (
#ifdef __CUDACC__
                        exp(out_minus_3[(k_39 * 1)])
#else
                        exp(out_minus_3[(k_39 * 1)])
#endif
                    );
                  }

                  // Finished code block for
                  // Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).
                }

                for (int k_40 = 0; k_40 < 1; k_40 += (1)) {
                  fout[(k_40 * 1)] = (
#ifdef __CUDACC__
                      ((yjrel + 3)[(k_40 * 1)] * out_exp_3[(k_40 * 1)])
#else
                      ((yjrel + 3)[(k_40 * 1)] * out_exp_3[(k_40 * 1)])
#endif
                  );
                }

                // Finished code block for
                // Var(4,1,0)*Exp(-(Var(0,1,2)*Sum((Var(1,3,0)-Var(2,3,1))**2))).
              }

              // Call the function, which outputs results in fout

              for (int k_41 = 0; k_41 < 1; k_41 += (1)) {
                tmp[(k_41 * 1)] += (fout[(k_41 * 1)]);
              }
            }
          }

          for (int k_42 = 0; k_42 < 1; k_42 += (1)) {
            acc[(k_42 * 1)] += (tmp[(k_42 * 1)]);
          }
        }
        __syncthreads();
      }
      if (index + 1 < end_slice)
        start_y = ranges_y[2 * index + 2];
    }
  }
  if (i < end_x) {

    for (int k_43 = 0; k_43 < 1; k_43 += (1)) {
      (out + i * 1)[k_43] = (acc[k_43]);
    }
  }
}
