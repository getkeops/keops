
#define C_CONTIGUOUS 1
#define USE_HALF 0

extern "C" __global__ void
GpuConv1DOnDevice_ranges(signed long int nx, signed long int ny, int nbatchdims,
                         signed long int *offsets_d, signed long int *lookup_d,
                         signed long int *slices_x, signed long int *ranges_y,
                         double *out, double **arg_0) {

  signed long int offsets[3];
  signed long int *indices_i = offsets;
  signed long int *indices_j = offsets + 1;
  signed long int *indices_p = offsets + 1 + 1;

  if (nbatchdims > 0)
    for (int k = 0; k < 3; k++)
      offsets[k] = offsets_d[3 * blockIdx.x + k];
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
  extern __shared__ double yj[];

  // load parameter(s)
  double param_loc[12];

  if (nbatchdims == 0) {
    {
      signed long int a = 0;

      for (signed long int v = 0; v < 12; v++) {
        param_loc[a] = arg_0[0][0 * 12 + v];
        a++;
      }
    }

  } else {
    {
      signed long int a = 0;

      for (signed long int v = 0; v < 12; v++) {
        param_loc[a] = arg_0[0][(0 + indices_p[0]) * 12 + v];
        a++;
      }
    }
  }
  double fout[1];

  double acc[1];
  double tmp[1];

  if (i < end_x) {
    for (int k_0 = 0; k_0 < 1; k_0 += (1)) {
      acc[k_0] = (double)(0.0f);
    }
    // acc = 0

    if (nbatchdims == 0) {
      {
        signed long int a = 0;
      }
      // load xi variables from global memory to local thread memory
    } else {
      { signed long int a = 0; }
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

              for (signed long int v = 0; v < 1; v++) {
                (yj + threadIdx.x * 1)[a] = arg_0[2][j * 1 + v];
                a++;
              }
            }
            // load yj variables from global memory to shared memory
          } else {
            {
              signed long int a = 0;

              for (signed long int v = 0; v < 1; v++) {
                (yj + threadIdx.x * 1)[a] =
                    arg_0[2][((j - start_y) + indices_j[0]) * 1 + v];
                a++;
              }
            }
            // Possibly, with offsets as we support broadcasting over batch
            // dimensions
          }
        __syncthreads();

        if (i < end_x) {      // we compute x1i only if needed
          double *yjrel = yj; // Loop on the columns of the current block.
          for (int k_1 = 0; k_1 < 1; k_1 += (1)) {
            tmp[k_1] = (double)(0.0f);
          }

          if (nbatchdims == 0) {
            for (signed long int jrel = 0;
                 (jrel < blockDim.x) && (jrel < end_y - jstart);
                 jrel++, yjrel += 1) {

              {
                // Starting code block for
                // Var(0,12,2)[(4*Var(1,1,0)+Var(2,1,1))].

                double out_add_0[1];

                {
                  // Starting code block for 4*Var(1,1,0)+Var(2,1,1).

                  double out_mult_0[1];

                  {
                    // Starting code block for 4*Var(1,1,0).

                    double out_intcst_0[1];

                    {
                      // Starting code block for 4.

                      *out_intcst_0 = (double)((float)4);

                      // Finished code block for 4.
                    }

                    for (int k_2 = 0; k_2 < 1; k_2 += (1)) {
                      out_mult_0[(k_2 * 1)] = (
#ifdef __CUDACC__
                          (out_intcst_0[(k_2 * 1)] *
                           (arg_0[1] + i * 1)[(k_2 * 1)])
#else
                          (out_intcst_0[(k_2 * 1)] *
                           (arg_0[1] + i * 1)[(k_2 * 1)])
#endif
                      );
                    }

                    // Finished code block for 4*Var(1,1,0).
                  }

                  for (int k_3 = 0; k_3 < 1; k_3 += (1)) {
                    out_add_0[(k_3 * 1)] =
                        out_mult_0[(k_3 * 1)] + (yjrel + 0)[(k_3 * 1)];
                  }

                  // Finished code block for 4*Var(1,1,0)+Var(2,1,1).
                }

                (*fout) = ((param_loc + 0)[(signed long int)(*out_add_0)]);

                // Finished code block for
                // Var(0,12,2)[(4*Var(1,1,0)+Var(2,1,1))].
              }

              // Call the function, which outputs results in xi[0:DIMX1]

              for (int k_4 = 0; k_4 < 1; k_4 += (1)) {
                tmp[(k_4 * 1)] += (fout[(k_4 * 1)]);
              }
            }
          } else {
            for (signed long int jrel = 0;
                 (jrel < blockDim.x) && (jrel < end_y - jstart);
                 jrel++, yjrel += 1) {

              {
                // Starting code block for
                // Var(0,12,2)[(4*Var(1,1,0)+Var(2,1,1))].

                double out_add_1[1];

                {
                  // Starting code block for 4*Var(1,1,0)+Var(2,1,1).

                  double out_mult_1[1];

                  {
                    // Starting code block for 4*Var(1,1,0).

                    double out_intcst_1[1];

                    {
                      // Starting code block for 4.

                      *out_intcst_1 = (double)((float)4);

                      // Finished code block for 4.
                    }

                    for (int k_5 = 0; k_5 < 1; k_5 += (1)) {
                      out_mult_1[(k_5 * 1)] = (
#ifdef __CUDACC__
                          (out_intcst_1[(k_5 * 1)] *
                           (arg_0[1] + (i + indices_i[0]) * 1)[(k_5 * 1)])
#else
                          (out_intcst_1[(k_5 * 1)] *
                           (arg_0[1] + (i + indices_i[0]) * 1)[(k_5 * 1)])
#endif
                      );
                    }

                    // Finished code block for 4*Var(1,1,0).
                  }

                  for (int k_6 = 0; k_6 < 1; k_6 += (1)) {
                    out_add_1[(k_6 * 1)] =
                        out_mult_1[(k_6 * 1)] + (yjrel + 0)[(k_6 * 1)];
                  }

                  // Finished code block for 4*Var(1,1,0)+Var(2,1,1).
                }

                (*fout) = ((param_loc + 0)[(signed long int)(*out_add_1)]);

                // Finished code block for
                // Var(0,12,2)[(4*Var(1,1,0)+Var(2,1,1))].
              }

              // Call the function, which outputs results in fout

              for (int k_7 = 0; k_7 < 1; k_7 += (1)) {
                tmp[(k_7 * 1)] += (fout[(k_7 * 1)]);
              }
            }
          }

          for (int k_8 = 0; k_8 < 1; k_8 += (1)) {
            acc[(k_8 * 1)] += (tmp[(k_8 * 1)]);
          }
        }
        __syncthreads();
      }
      if (index + 1 < end_slice)
        start_y = ranges_y[2 * index + 2];
    }
  }
  if (i < end_x) {

    for (int k_9 = 0; k_9 < 1; k_9 += (1)) {
      (out + i * 1)[k_9] = (acc[k_9]);
    }
  }
}
