/*
 * Copyright 2022 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PIPELINE_LSTM_HEADER_H
#define PIPELINE_LSTM_HEADER_H //These are guards whihc prevent includion of multiple files.

#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

#include "activations.hpp"
#include "eltwise.hpp"
#include "mvau.hpp"
#include "weights.hpp"
#include "mac.hpp"
#include "utils.hpp"



//-------------------------------------------------------------------------------------

constexpr unsigned Inp_N = 10; //This factor defines the input length at each time step.
constexpr unsigned Out_N = 20; //This factor defines the number of LSTM cells in the LSTM layer.
constexpr unsigned Act_N_255 = 255;// This factor defines the number of thresholds or quantization level (2 ^^ n - 1)
constexpr unsigned Act_N_63 = 63;
constexpr unsigned Act_N_62 = 62;
constexpr unsigned dense_1 = 64;
constexpr unsigned dense_2 = 32;
constexpr unsigned dense_3 = 5;
constexpr unsigned num_lstm_steps = 2;
constexpr unsigned num_test_inputs = 3;

using qlstm_f32_t = float;
using qlstm_uint6_t = ap_uint<6>;
using qlstm_int6_t = ap_int<6>;
using qlstm_int7_t = ap_int<7>;
using qlstm_int8_t = ap_int<8>;
using qlstm_int9_t = ap_int<9>;
using qlstm_uint8_t = ap_uint<8>;
using qlstm_uint32_t = ap_uint<32>;
using qlstm_int32_t = ap_int<32>;

void qlstm_top_2(
		//inputs
		hls::stream<ap_axis<32,2,5,6>>& x_input_final,
		//output streams
		hls::stream<ap_axis<32,2,5,6>>& final_output
);

//Updated
void initializeThresholds_1(const qlstm_int32_t mt_3[Out_N][Act_N_62],const qlstm_int32_t mt_4[Out_N][Act_N_62],
							const qlstm_int32_t mt_5[Out_N][Act_N_62],const qlstm_int32_t mt_6[Out_N][Act_N_62],
							const qlstm_int32_t mt_19[dense_1][Act_N_255],const qlstm_int32_t mt_20[dense_2][Act_N_255]);

void initializeThresholds_2(
		const qlstm_f32_t  mt_0[Act_N_255],const qlstm_f32_t  mt_1[Act_N_255],const qlstm_f32_t  mt_2[Act_N_63],
		const qlstm_int8_t mt_7[Act_N_63],const qlstm_int8_t mt_8[Act_N_63],const qlstm_int8_t  mt_9[Act_N_63],const qlstm_int8_t  mt_10[Act_N_63],
		const qlstm_int32_t mt_11[Act_N_62],const qlstm_int32_t mt_12[Act_N_62],
		const qlstm_int32_t mt_13[Act_N_62],const qlstm_int32_t mt_14[Act_N_62],
		const qlstm_int8_t mt_15[Act_N_63],
		const qlstm_int32_t mt_16[Act_N_255],const qlstm_int32_t mt_17[Act_N_255],
		const qlstm_int8_t mt_18[Act_N_255]
		);

//Updated
void initializeWeights(
		const qlstm_int8_t mm_weights_0[Out_N][Inp_N],const qlstm_int8_t mm_weights_1[Out_N][Inp_N],const qlstm_int8_t mm_weights_2[Out_N][Inp_N],const qlstm_int8_t mm_weights_3[Out_N][Inp_N],
		const qlstm_int8_t mm_weights_4[Out_N][Out_N],const qlstm_int8_t mm_weights_5[Out_N][Out_N],const qlstm_int8_t mm_weights_6[Out_N][Out_N],const qlstm_int8_t mm_weights_7[Out_N][Out_N],
		const qlstm_int6_t mm_weights_8[dense_1][Out_N],const qlstm_int6_t mm_weights_9[dense_2][dense_1],const qlstm_int6_t mm_weights_10[dense_3][dense_2]
		);

#endif












//void qlstm_top(hls::stream<qlstm_int8_t>& inStream,
//        hls::stream<qlstm_int8_t>& outStream,
//		hls::stream<qlstm_int8_t>& in0, hls::stream<qlstm_int8_t>& in1,
//		hls::stream<qlstm_int8_t>& in0_elmul, hls::stream<qlstm_int8_t>& in1_elmul,
//		hls::stream<qlstm_int32_t>& out0, hls::stream<qlstm_int32_t>& out_elmul,
//		hls::stream<qlstm_int8_t>& mm_in0, ap_int<8> mm_weights[20][10],
//		hls::stream<qlstm_int32_t>& mm_output,
//		hls::stream<qlstm_int32_t>& mul_in,
//		hls::stream<float>& mul_out
//		);
