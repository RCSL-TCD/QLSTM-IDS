
/******************************************************************************
 *	@brief
 *		Templatized streaming elementwise operator, consuming two streams and
 *		producing a third. The operator and datatypes can be customized.
 *	@author Yaman Umuroglu <yamanu@amd.com>
 ******************************************************************************/

#ifndef ELTWISE_HPP
#define ELTWISE_HPP

#include <hls_stream.h>


/**
 * \brief StreamingEltwise function
 *
 * The function performs a generic eltwise function on two streams and
 * produces an output stream.
 *
 * \tparam Channels   Number of channels for eltwise operation
 * \tparam PE         Number of channels for eltwise operation computed in parallel
 * \tparam N          Total number of elements to process
 * \tparam SliceIn0   Data slicer for input 0 type
 * \tparam SliceIn1   Data slicer for input 1 type
 * \tparam SliceOut   Data slicer for output type
 * \tparam TStrmIn0   Type of the input 0 stream - safely deducible from the paramaters
 * \tparam TStrmIn1   Type of the input 1 stream - safely deducible from the paramaters
 * \tparam TStrmOut   Type of the output - safely deducible from the paramaters
 * \tparam TFxn       Type of the function class (e.g. Max, Avg, Sum) - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param function    Function to apply, derived from EltwiseFunction
 */
template<
	unsigned Channels, unsigned PE, unsigned N,
	typename SliceIn0, typename SliceIn1, typename SliceOut,
	typename TStrmIn0, typename TStrmIn1, typename TStrmOut,
	typename Fxn
>
void StreamingEltwise(
	hls::stream<TStrmIn0> &in0,
	hls::stream<TStrmIn1> &in1,
	hls::stream<TStrmOut> &out,
	Fxn &&f
) {
	constexpr unsigned  TOTAL_FOLD = (Channels / PE) * N;
	// everything merged into a common iteration space (one big loop instead
	// of smaller nested loops) to get the pipelining the way we want
	for(unsigned  i = 0; i < TOTAL_FOLD; i++) {
#pragma HLS pipeline style=flp II=1
		auto const  in0_slice_channels = SliceIn0()(in0.read(), 0);
		auto const  in1_slice_channels = SliceIn1()(in1.read(), 0);
		auto outElem = SliceOut().template operator()<TStrmOut>();
		for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
			outElem(pe, 0, 1) = f(in0_slice_channels(pe, 0), in1_slice_channels(pe, 0));
		}
		out.write(outElem);
	}
};

#endif
