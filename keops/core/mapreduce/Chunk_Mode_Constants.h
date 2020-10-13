#pragma once


namespace keops {

template < class FUN >
struct Chunk_Mode_Constants {

	static const int DIMRED = FUN::DIMRED; // dimension of reduction operation

	typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
	typedef typename FUN::INDSP INDSP;
	static const int DIMP = DIMSP::SUM;

	static const int DIMOUT = FUN::DIM; // dimension of output variable
	static const int DIMFOUT = FUN::F::DIM; // dimension of output variable

	static const int DIM_ORG = FUN::F::template CHUNKED_FORMULAS<DIMCHUNK>::FIRST::NEXT::FIRST::FIRST;
	static const int NCHUNKS = 1 + (DIM_ORG-1) / DIMCHUNK;
	static const int DIMLASTCHUNK = DIM_ORG - (NCHUNKS-1)*DIMCHUNK;
	static const int NMINARGS = FUN::NMINARGS;

	using FUN_CHUNKED = typename FUN::F::template CHUNKED_FORMULAS<DIMCHUNK>::FIRST::FIRST;
	static const int DIMOUT_CHUNK = FUN_CHUNKED::DIM;
	using VARSI_CHUNKED = typename FUN_CHUNKED::template CHUNKED_VARS<FUN::tagI>;
	using DIMSX_CHUNKED = GetDims<VARSI_CHUNKED>;
	using INDSI_CHUNKED = GetInds<VARSI_CHUNKED>;
	using VARSJ_CHUNKED = typename FUN_CHUNKED::template CHUNKED_VARS<FUN::tagJ>;
	using DIMSY_CHUNKED = GetDims<VARSJ_CHUNKED>;
	using INDSJ_CHUNKED = GetInds<VARSJ_CHUNKED>;

	using FUN_POSTCHUNK  = typename FUN::F::template POST_CHUNK_FORMULA < NMINARGS >;
	using VARSI_POSTCHUNK = typename FUN_POSTCHUNK::template VARS<FUN::tagI>;
	using DIMSX_POSTCHUNK = GetDims<VARSI_POSTCHUNK>;
	using VARSJ_POSTCHUNK = typename FUN_POSTCHUNK::template VARS<FUN::tagJ>;
	using DIMSY_POSTCHUNK = GetDims<VARSJ_POSTCHUNK>;

	using VARSI_NOTCHUNKED = MergePacks < VARSI_POSTCHUNK, typename FUN_CHUNKED::template NOTCHUNKED_VARS<FUN::tagI> >;
	using INDSI_NOTCHUNKED = GetInds<VARSI_NOTCHUNKED>;
	using DIMSX_NOTCHUNKED = GetDims<VARSI_NOTCHUNKED>;
	static const int DIMX_NOTCHUNKED = DIMSX_NOTCHUNKED::SUM;

	using VARSJ_NOTCHUNKED = MergePacks < VARSJ_POSTCHUNK, typename FUN_CHUNKED::template NOTCHUNKED_VARS<FUN::tagJ> >;
	using INDSJ_NOTCHUNKED = GetInds<VARSJ_NOTCHUNKED>;
	using DIMSY_NOTCHUNKED = GetDims<VARSJ_NOTCHUNKED>;
	static const int DIMY_NOTCHUNKED = DIMSY_NOTCHUNKED::SUM;

	using FUN_LASTCHUNKED = typename FUN::F::template CHUNKED_FORMULAS<DIMLASTCHUNK>::FIRST::FIRST;
	using VARSI_LASTCHUNKED = typename FUN_LASTCHUNKED::template CHUNKED_VARS<FUN::tagI>;
	using DIMSX_LASTCHUNKED = GetDims<VARSI_LASTCHUNKED>;
	using VARSJ_LASTCHUNKED = typename FUN_LASTCHUNKED::template CHUNKED_VARS<FUN::tagJ>;
	using DIMSY_LASTCHUNKED = GetDims<VARSJ_LASTCHUNKED>;

	using VARSI = ConcatPacks < VARSI_NOTCHUNKED, VARSI_CHUNKED >;
	using DIMSX = GetDims<VARSI>;
	using INDSI = GetInds<VARSI>;
	static const int DIMX = DIMSX::SUM;

	using VARSJ = ConcatPacks < VARSJ_NOTCHUNKED, VARSJ_CHUNKED >;
	using DIMSY = GetDims<VARSJ>;
	using INDSJ = GetInds<VARSJ>;
	static const int DIMY = DIMSY::SUM;

	using INDS = ConcatPacks < ConcatPacks < INDSI, INDSJ >, INDSP >;

	using VARSI_LAST = ConcatPacks < VARSI_NOTCHUNKED, VARSI_LASTCHUNKED >;
	using DIMSX_LAST = GetDims<VARSI_LAST>;

	using VARSJ_LAST = ConcatPacks < VARSJ_NOTCHUNKED, VARSJ_LASTCHUNKED >;
	using DIMSY_LAST = GetDims<VARSJ_LAST>;
};

template < class FUN, int USE_CHUNK_MODE > struct Get_DIMY_SHARED;

template < class FUN >
struct Get_DIMY_SHARED<FUN,0> {
    static const int Value = FUN::DIMSY::SUM;
};

template < class FUN >
struct Get_DIMY_SHARED<FUN,1> {
    static const int Value = Chunk_Mode_Constants<FUN>::DIMY;
};

template < class FUN >
struct Get_DIMY_SHARED<FUN,2> {
    static const int Value = DIMFINALCHUNK;
};

}
