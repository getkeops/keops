from keops.utils.code_gen_utils import Var_loader


def define_broadcast_index_function():
    return """
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
            """


def define_vect_broadcast_index_function():
    return """
                    void vect_broadcast_index(int i,
                                              int nbatchdims,
                                              int nvars,
                                              int *full_shape,
                                              int *reduced_shapes,
                                              int *out,
                                              int add_offset = 0) {
                      for (int k = 0; k < nvars; k++) {
                        out[k] = add_offset + broadcast_index(i, nbatchdims, full_shape, reduced_shapes + (nbatchdims + 1) * k);
                      }
                    }
                """


def define_fill_shapes_function(red_formula):
    varloader = Var_loader(red_formula)
    size_i = len(varloader.Varsi) + 1
    size_j = len(varloader.Varsj)
    size_p = len(varloader.Varsp)
    inds_i = varloader.indsi
    inds_j = varloader.indsj
    inds_p = varloader.indsp
    tagIJ = (
        red_formula.tagJ
    )  # 1 if the reduction is made "over j", 0 if it is made "over i"

    string = """
                void fill_shapes(int nbatchdims, int *shapes, int *shapes_i, int *shapes_j, int *shapes_p) {

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
            """
    for k in range(size_i - 1):
        string += f"""
                      for (int l = 0; l < nbatchdims; l++) {{  // l-th column
                        shapes_i[{k} * (nbatchdims + 1) + l] = shapes[(1 + {inds_i[k]}) * (nbatchdims + 3) + l];
                      }}  
                      shapes_i[{k} * (nbatchdims + 1) + nbatchdims] =
                          shapes[(1 + {inds_i[k]}) * (nbatchdims + 3) + nbatchdims + 1 - {tagIJ}];

                          
                      // Then, we do the same for shapes_j, but with "N" instead of "M":
                """

    for k in range(size_j):
        string += f"""         
                        for (int l = 0; l < nbatchdims; l++) {{  // l-th column
                          shapes_j[{k} * (nbatchdims + 1) + l] = shapes[(1 + {inds_j[k]}) * (nbatchdims + 3) + l];
                        }}
                        shapes_j[{k} * (nbatchdims + 1) + nbatchdims] = shapes[(1 + {inds_j[k]}) * (nbatchdims + 3) + nbatchdims + {tagIJ}];
                  

                        // And finally for the parameters, with "1" instead of "M":
                  """

    for k in range(size_p):
        string += f"""  
                        for (int l = 0; l < nbatchdims; l++) {{  // l-th column
                          shapes_p[{k} * (nbatchdims + 1) + l] = shapes[(1 + {inds_p[k]}) * (nbatchdims + 3) + l];
                        }}
                        shapes_p[{k} * (nbatchdims + 1) + nbatchdims] = 1;

                   """

    string += """
                    }
              """

    return string
