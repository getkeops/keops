#pragma once

namespace keops {

/////////////////////////////////////////////////////////////////////////////////
//                           Keops                                             //
/////////////////////////////////////////////////////////////////////////////////

using FF = F::F; // F::F is formula inside reduction (ex if F is SumReduction<Form> then F::F is Form)

using VARSI = typename FF::template VARS< 0 >;    // list variables of type I used in formula F
using VARSJ = typename FF::template VARS< 1 >;    // list variables of type J used in formula F
using VARSP = typename FF::template VARS< 2 >;    // list variables of type parameter used in formula F

using DIMSX = GetDims< VARSI >;
using DIMSY = GetDims< VARSJ >;
using DIMSP = GetDims< VARSP >;

using INDSI = GetInds< VARSI >;
using INDSJ = GetInds< VARSJ >;
using INDSP = GetInds< VARSP >;

using INDS = ConcatPacks <ConcatPacks< INDSI, INDSJ >, INDSP>;

const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F

const int NMINARGS = F::NMINARGS;

const int DIMOUT = F::DIM;

const int TAGIJ = F::tagI;

const std::string f = PrintReduction< F >();

}
