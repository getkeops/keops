#pragma once
#include "Sizes.h"

template < typename TYPE >
class Ranges {
  public:
    int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y;

    int **castedranges;
    int *ranges_i, *slices_i, *redranges_j;

    Ranges(Sizes<TYPE> sizes, int **ranges) {

        // Sparsity: should we handle ranges? ======================================
        if (sizes.nbatchdims == 0) {  // Standard M-by-N computation
            if (ranges[6][0] == -1) {
                tagRanges = 0;

                nranges_x = 0;
                nranges_y = 0;

                nredranges_x = 0;
                nredranges_y = 0;

            } else {
                tagRanges = 1;
                nranges_x = ranges[6][0];
                nranges_y = ranges[6][3];
                nredranges_x = ranges[6][5];
                nredranges_y = ranges[6][2];

                // get the pointers to data to avoid a copy
                castedranges = (int**) malloc(sizeof(int*)*6);
                for (int i = 0; i < 6; i++) {
                    castedranges[i] = ranges[i];
                }
            }

        } else if (ranges[6][0] == -1) {
            // Batch processing: we'll have to generate a custom, block-diagonal sparsity pattern
            tagRanges = 1;  // Batch processing is emulated through the block-sparse mode

            // Create new "castedranges" from scratch ------------------------------
            // With pythonic notations, we'll have:
            //   castedranges = (ranges_i, slices_i, redranges_j,   ranges_j, slices_j, redranges_i)
            // with:
            // - ranges_i    = redranges_i = [ [0,M], [M,2M], ..., [(nbatches-1)M, nbatches*M] ]
            // - slices_i    = slices_j    = [    1,     2,   ...,   nbatches-1,   nbatches    ]
            // - redranges_j = ranges_j    = [ [0,N], [N,2N], ..., [(nbatches-1)N, nbatches*N] ]

            //int* castedranges[6];
            castedranges = (int**) malloc(sizeof(int*)*6);

            //int ranges_i[2 * sizes.nbatches];  // ranges_i
            ranges_i = (int*) malloc(sizeof(int)*(2 * sizes.nbatches));
            for(int i=0; i<2 * sizes.nbatches; i++)
                ranges_i[i] = 0;

            //int slices_i[sizes.nbatches];    // slices_i
            slices_i = (int*) malloc(sizeof(int)*(sizes.nbatches));
            for(int i=0; i<sizes.nbatches; i++)
                slices_i[i] = 0;

            //int redranges_j[2 * sizes.nbatches];  // redranges_j
            redranges_j = (int*) malloc(sizeof(int)*(2 * sizes.nbatches));
            for(int i=0; i<2 * sizes.nbatches; i++)
                redranges_j[i] = 0;
            for (int b = 0; b < sizes.nbatches; b++) {
                ranges_i[2 * b] = b * sizes.M;
                ranges_i[2 * b + 1] = (b + 1) * sizes.M;
                slices_i[b] = (b + 1);
                redranges_j[2 * b] = b * sizes.N;
                redranges_j[2 * b + 1] = (b + 1) * sizes.N;
            }

            castedranges[0] = &ranges_i[0];
            castedranges[1] = &slices_i[0];
            castedranges[2] = &redranges_j[0];
            castedranges[3] = &redranges_j[0];            // ranges_j
            castedranges[4] = &slices_i[0];            // slices_j
            castedranges[5] = &ranges_i[0];            // redranges_i

            nranges_x = sizes.nbatches;
            nredranges_x = sizes.nbatches;
            nranges_y = sizes.nbatches;
            nredranges_y = sizes.nbatches;

        }
#if do_checks
        else {
            throw std::runtime_error(
                "[KeOps] The 'ranges' argument (block-sparse mode) is not supported with batch processing, "
                "but we detected " + std::to_string(sizes.nbatchdims) + " > 0 batch dimensions."
            );
        }
#endif



    };

};
