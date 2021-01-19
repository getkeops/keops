extern "C" int CpuReduc(int nx, int ny, {TYPE}* gamma, {TYPE}** args) {{
#pragma omp parallel for
    for (int i = 0; i < nx; i++) {{
    {TYPE} fout[{DIMFOUT}], xi[{DIMX}], yj[{DIMY}];
    {TYPEACC} acc[DIMRED];
    {loadx} // load< DIMSX, INDSI >(i, xi, args);
      {InitializeReduction} // typename FUN::template InitializeReduction< TYPEACC, TYPE >()(acc);   // acc = 0
      for (int j = 0; j < ny; j++) {{
        {loady} // load< DIMSY, INDSJ >(j, yj, args);
        {call} // call< DIMSX, DIMSY, DIMSP >(fun, fout, xi, yj, pp);
        {ReducePairShort} // typename FUN::template ReducePairShort< TYPEACC, TYPE >()(acc, fout, j); // acc += fout
      }}
        {FinalizeOutput} // typename FUN::template FinalizeOutput< TYPEACC, TYPE >()(acc, out + i * DIMOUT, i);
    }}
    return 0;
}}

