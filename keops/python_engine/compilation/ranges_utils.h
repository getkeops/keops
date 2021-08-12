#pragma once


int broadcast_index(int i, int nbatchdims, int *full_shape, int *shape) {
    int M_N = shape[nbatchdims];
    int res = i % M_N, step = M_N, full_step = M_N;
    for (int b = nbatchdims; b > 0; b--) {
        if (shape[b - 1] != 1) {
            res += ((i / full_step) % shape[b - 1]) * step;
        }
        full_step *= full_shape[b - 1];
        step *= shape[b - 1];
    }
    return res;
}

void vect_broadcast_index(int i, int nbatchdims, int nvars, int *full_shape,
                          int *reduced_shapes, int *out, int add_offset = 0) {
    for (int k = 0; k < nvars; k++) {
        out[k] = add_offset + broadcast_index(i, nbatchdims, full_shape, reduced_shapes + (nbatchdims + 1) * k);
    }
}

void fill_shapes(int nbatchdims, int *shapes, int *shapes_i, int *shapes_j, int *shapes_p,
                 int tagJ, int sizei, int sizej, int sizep, int *indsi, int *indsj, int *indsp) {

    sizei += 1;

    const int tagIJ = tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"

    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // N.B.: If tagIJ == 1, the reduction is made over 'j', which is the default mode.
    //       However, if tagIJ == 0, the reduction is performed over the 'i' variables:
    //       since "shape" does not change, we must adapt the adress at which we pick information...
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

    // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
    // making it look like, say:
    // [ A, .., B, M]
    // [ A, .., 1, M]
    // [ A, .., A, M]
    for (int k = 0; k < (sizei - 1); k++) {  // k-th line
        for (int l = 0; l < nbatchdims; l++) {  // l-th column
            shapes_i[k * (nbatchdims + 1) + l] = shapes[(1 + get_val(indsi, k)) * (nbatchdims + 3) + l];
        }
        shapes_i[k * (nbatchdims + 1) + nbatchdims] =
                shapes[(1 + get_val(indsi, k)) * (nbatchdims + 3) + nbatchdims + 1 - tagIJ];
    }

    // Then, we do the same for shapes_j, but with "N" instead of "M":
    for (int k = 0; k < sizej; k++) {  // k-th line
        for (int l = 0; l < nbatchdims; l++) {  // l-th column
            shapes_j[k * (nbatchdims + 1) + l] = shapes[(1 + get_val(indsj, k)) * (nbatchdims + 3) + l];
        }
        shapes_j[k * (nbatchdims + 1) + nbatchdims] = shapes[(1 + get_val(indsj, k)) * (nbatchdims + 3) + nbatchdims +
                                                             tagIJ];
    }

    // And finally for the parameters, with "1" instead of "M":
    for (int k = 0; k < sizep; k++) {  // k-th line
        for (int l = 0; l < nbatchdims; l++) {  // l-th column
            shapes_p[k * (nbatchdims + 1) + l] = shapes[(1 + get_val(indsp, k)) * (nbatchdims + 3) + l];
        }
        shapes_p[k * (nbatchdims + 1) + nbatchdims] = 1;
    }

}


int *build_offset_tables(int nbatchdims, int *shapes, int nblocks, __INDEX__ *lookup_h,
                         int sizei, int sizej, int sizep, int *indsi, int *indsj, int *indsp,
                         int tagJ) {

    // Support for broadcasting over batch dimensions =============================================

    int sizevars = sizei + sizej + sizep;

    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

    int shapes_i[sizei * (nbatchdims + 1)], shapes_j[sizej * (nbatchdims + 1)], shapes_p[sizep * (nbatchdims + 1)];

    // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
    // making it look like, say:
    // [ A, .., B, M]
    // [ A, .., 1, M]
    // [ A, .., A, M]
    // Then, we do the same for shapes_j, but with "N" instead of "M".
    // And finally for the parameters, with "1" instead of "M".
    fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p, tagJ, sizei, sizej, sizep, indsi, indsj, indsp);

    int tagIJ = tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"
    int M = shapes[nbatchdims], N = shapes[nbatchdims + 1];

    // We create a lookup table, "offsets", of shape (nblocks, SIZEVARS) --------
    int *offsets_h = NULL, *offsets_d = NULL;

    offsets_h = new int[nblocks * sizevars];

    for (int k = 0; k < nblocks; k++) {
        int range_id = (int) lookup_h[3 * k];
        int start_x = tagIJ ? range_id * M : range_id * N;
        int start_y = tagIJ ? range_id * N : range_id * M;

        int patch_offset = (int) (lookup_h[3 * k + 1] - start_x);

        vect_broadcast_index(start_x, nbatchdims, sizei, shapes, shapes_i, offsets_h + k * sizevars, patch_offset);
        vect_broadcast_index(start_y, nbatchdims, sizej, shapes, shapes_j, offsets_h + k * sizevars + sizei);
        vect_broadcast_index(range_id, nbatchdims, sizep, shapes, shapes_p, offsets_h + k * sizevars + sizei + sizej);
    }

    cuMemAlloc((CUdeviceptr *) &offsets_d, sizeof(int) * nblocks * sizevars);
    cuMemcpyHtoD((CUdeviceptr) offsets_d, offsets_h, sizeof(int) * nblocks * sizevars);

    delete[] offsets_h;
    return offsets_d;
}


void range_preprocess(int tagHostDevice, int &nblocks, int tagI, int nranges_x, int nranges_y, __INDEX__ **castedranges,
                      int nbatchdims, __INDEX__ *&slices_x_d, __INDEX__ *&ranges_y_d,
                      __INDEX__ *&lookup_d, int *&offsets_d, int blockSize_x,
                      int *indsi, int *indsj, int *indsp, int *shapes) {

    // Ranges pre-processing... ==================================================================

    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h:
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j

    int tagJ = 1 - tagI;
    int nranges = tagJ ? nranges_x : nranges_y;

    __INDEX__ *ranges_x = tagJ ? castedranges[0] : castedranges[3];
    __INDEX__ *slices_x = tagJ ? castedranges[1] : castedranges[4];
    __INDEX__ *ranges_y = tagJ ? castedranges[2] : castedranges[5];

    __INDEX__ *ranges_x_h = NULL;

    // The code below needs a pointer to ranges_x on *host* memory,  -------------------
    // as well as pointers to slices_x and ranges_y on *device* memory.
    // -> Depending on the "ranges" location, we'll copy ranges_x *or* slices_x and ranges_y
    //    to the appropriate memory:
    bool ranges_on_device = ((tagHostDevice == 1) && nbatchdims == 0);
    // N.B.: We only support Host ranges with Device data when these ranges were created
    //       to emulate block-sparse reductions.

    if (ranges_on_device) {  // The ranges are on the device
        ranges_x_h = new __INDEX__[2 * nranges];
        // Send data from device to host.
        cuMemcpyDtoH(ranges_x_h, (CUdeviceptr) ranges_x, sizeof(__INDEX__) * 2 * nranges);
        slices_x_d = slices_x;
        ranges_y_d = ranges_y;
    } else {  // The ranges are on host memory; this is typically what happens with **batch processing**,
        // with ranges generated by keops_io.h:
        ranges_x_h = ranges_x;
        // Copy "slices_x" to the device:

        cuMemAlloc((CUdeviceptr *) &slices_x_d, sizeof(__INDEX__) * nranges);
        cuMemcpyHtoD((CUdeviceptr) slices_x_d, slices_x, sizeof(__INDEX__) * nranges);

        // Copy "redranges_y" to the device: with batch processing, we KNOW that they have the same shape as ranges_x
        cuMemAlloc((CUdeviceptr *) &ranges_y_d, sizeof(__INDEX__) * 2 * nranges);
        cuMemcpyHtoD((CUdeviceptr) ranges_y_d, ranges_y, sizeof(__INDEX__) * 2 * nranges);
    }

    // Computes the number of blocks needed ---------------------------------------------
    nblocks = 0;
    int len_range = 0;
    for (int i = 0; i < nranges; i++) {
        len_range = ranges_x_h[2 * i + 1] - ranges_x_h[2 * i];
        nblocks += (len_range / blockSize_x) + (len_range % blockSize_x == 0 ? 0 : 1);
    }

    // Create a lookup table for the blocks --------------------------------------------
    __INDEX__ *lookup_h = NULL;
    lookup_h = new __INDEX__[3 * nblocks];
    int index = 0;
    for (int i = 0; i < nranges; i++) {
        len_range = ranges_x_h[2 * i + 1] - ranges_x_h[2 * i];
        for (int j = 0; j < len_range; j += blockSize_x) {
            lookup_h[3 * index] = i;
            lookup_h[3 * index + 1] = ranges_x_h[2 * i] + j;
            lookup_h[3 * index + 2] = ranges_x_h[2 * i] + j + ::std::min((int) blockSize_x, len_range - j);
            index++;
        }
    }

    // Load the table on the device -----------------------------------------------------
    cuMemAlloc((CUdeviceptr *) &lookup_d, sizeof(__INDEX__) * 3 * nblocks);
    cuMemcpyHtoD((CUdeviceptr) lookup_d, lookup_h, sizeof(__INDEX__) * 3 * nblocks);


    // Support for broadcasting over batch dimensions =============================================

    // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):

    int sizei = indsi[0];
    int sizej = indsj[0];
    int sizep = indsp[0];

    if (nbatchdims > 0) {
        offsets_d = build_offset_tables(nbatchdims, shapes, nblocks, lookup_h,
                                        sizei, sizej, sizep, indsi, indsj, indsp, tagJ);
    }


}
