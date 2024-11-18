#include "pipeline-lstm-header.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "ap_axi_sdata.h"
#include "hls_stream.h"

using namespace std;

int main(){

//Output HLS streams
	hls::stream<qlstm_f32_t> global_out;
	hls::stream<qlstm_f32_t> global_out1;
	hls::stream<ap_axis<32,2,5,6>> final_output;

	//Weigth Matrices
	 qlstm_int8_t mm_weights_0[Out_N][Inp_N];
	 qlstm_int8_t mm_weights_1[Out_N][Inp_N];
	 qlstm_int8_t mm_weights_2[Out_N][Inp_N];
	 qlstm_int8_t mm_weights_3[Out_N][Inp_N];
	//Recurrence Matrices
	 qlstm_int8_t mm_weights_4[Out_N][Out_N];
	 qlstm_int8_t mm_weights_5[Out_N][Out_N];
	 qlstm_int8_t mm_weights_6[Out_N][Out_N];
	 qlstm_int8_t mm_weights_7[Out_N][Out_N];
	//Dense matmul matrices
	 qlstm_int6_t mm_weights_8[dense_1][Out_N];
	 qlstm_int6_t mm_weights_9[dense_2][dense_1];
	 qlstm_int6_t mm_weights_10[dense_3][dense_2];

	//Thresholds for Multithresholding
	 qlstm_f32_t  mt_weights_0[Act_N_255], mt_weights_1[Act_N_255], mt_weights_2[Act_N_62];
	 qlstm_int32_t mt_weights_3[Out_N][Act_N_62], mt_weights_4[Out_N][Act_N_62], mt_weights_5[Out_N][Act_N_62], mt_weights_6[Out_N][Act_N_62];
	//Max value in the thresholds 32 i.e why threshold type int7.
	 qlstm_int8_t mt_weights_7[Act_N_63], mt_weights_8[Act_N_63],  mt_weights_9[Act_N_63],  mt_weights_10[Act_N_63];
	 qlstm_int32_t mt_weights_11[Act_N_62], mt_weights_12[Act_N_62], mt_weights_13[Act_N_62],mt_weights_14[Act_N_62];
	 qlstm_int8_t mt_weights_15[Act_N_63];
	 qlstm_int32_t mt_weights_16[Act_N_255],mt_weights_17[Act_N_255];
	 qlstm_int8_t mt_weights_18[Act_N_255];
	 qlstm_int32_t mt_weights_19[dense_1][Act_N_255], mt_weights_20[dense_2][Act_N_255];
	
	hls::stream<ap_axis<32,2,5,6>> x_input_final;
	qlstm_f32_t h0[Out_N];
	qlstm_f32_t c0[Out_N];

	ifstream inp_file;
	inp_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/hls_test_x_int5.txt");
	//Reading test inputs in the below loop line by line and storing it in an input test array.
	string inp_line0,inp_val0;
	int inp_row_mm=0;
		 while (getline(inp_file, inp_line0) && inp_row_mm < num_test_inputs) {
			 	 istringstream inp_iss0(inp_line0);
			 	 int inp_col = 0;
		         while (getline(inp_iss0, inp_val0, ',')  && inp_col < num_test_inputs*20) { //Within a line, reading values one by one with ',' as delimiter.
		        	 ap_axis<32,2,5,6> data_inp;
		        	 data_inp.data = stoi(inp_val0);
		        	 x_input_final.write(data_inp);
		        	 inp_col++;
		         }
		        inp_row_mm++;
		 }


//Initializing initial hidden state values with zeros in the loop.
	for (int k = 0; k < Out_N; ++k){
					h0[k] = 1;
					c0[k] = 1;
			}

	//----------  Load weight matrix values for matmuls -----------------------

	 ifstream mm_0_file,mm_1_file,mm_2_file,mm_3_file,mm_4_file,mm_5_file,mm_6_file,mm_7_file,mm_8_file,mm_9_file,mm_10_file;
	 mm_0_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_0_weights.txt");
	 mm_1_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_1_weights.txt");
	 mm_2_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_2_weights.txt");
	 mm_3_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_3_weights.txt");
	 mm_4_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_4_weights.txt");
	 mm_5_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_5_weights.txt");
     mm_6_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_6_weights.txt");
	 mm_7_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_7_weights.txt");
	 mm_8_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_8_weights.txt");
	 mm_9_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_9_weights.txt");
	 mm_10_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mm_10_weights.txt");
	 //Checking if all the files opened here.
	 if (!mm_0_file.is_open() || !mm_1_file.is_open() || !mm_2_file.is_open() || !mm_3_file.is_open() ||
	     !mm_4_file.is_open() || !mm_5_file.is_open() || !mm_6_file.is_open() || !mm_7_file.is_open() ||
		 !mm_8_file.is_open() || !mm_9_file.is_open() || !mm_10_file.is_open()) {
	         std::cerr << "Error opening some file" << std::endl;
	         return 1;
	     }
	 string line0,val0,line1,val1,line2,val2,line3,val3,line4,val4,line5,val5,line6,val6,line7,val7,line8,val8,line9,val9,line10,val10;
	 int row_mm=0;
	 while (getline(mm_0_file, line0) && getline(mm_1_file, line1) && getline(mm_2_file, line2) && getline(mm_3_file, line3) &&
			getline(mm_4_file, line4) && getline(mm_5_file, line5) && getline(mm_6_file, line6) && getline(mm_7_file, line7) && row_mm < Out_N) {
		 	 istringstream iss0(line0),iss1(line1),iss2(line2),iss3(line3),iss4(line4),iss5(line5),iss6(line6),iss7(line7);
		 	 int col = 0, col2 = 0;
	         while (getline(iss0, val0, ',') && getline(iss1, val1, ',') && getline(iss2, val2, ',') && getline(iss3, val3, ',') && col < Inp_N) { //Within a line, reading values one by one with ',' as delimiter.
	        	 mm_weights_0[row_mm][col] = stoi(val0); //stoi() converts strings to integers
	        	 //cout << "MM_Weight -------" << stoi(val0) << " -- " << col << std::endl;
	        	 mm_weights_1[row_mm][col] = stoi(val1);
	        	 mm_weights_2[row_mm][col] = stoi(val2);
	        	 mm_weights_3[row_mm][col] = stoi(val3);
	             col++;
	         }
	         while (getline(iss4, val4, ',') && getline(iss5, val5, ',') && getline(iss6, val6, ',') && getline(iss7, val7, ',') && col2 < Out_N) { //Within a line, reading values one by one with ',' as delimiter.
	      	     mm_weights_4[row_mm][col2] = stoi(val4);
	      	     mm_weights_5[row_mm][col2] = stoi(val5);
	      	     mm_weights_6[row_mm][col2] = stoi(val6);
	      	     mm_weights_7[row_mm][col2] = stoi(val7);
	      	     col2++;
	      	         }
	         row_mm++;
	     }
	 //Loading dense matmul weights
	 int row1_mm =0;
	 while (getline(mm_8_file, line8) && row1_mm < dense_1) {
		 istringstream iss8(line8);
		 int col3 = 0;
		 while (getline(iss8, val8, ',')  && col3 < Out_N) { //Within a line, reading values one by one with ',' as delimiter.
		 	        	 mm_weights_8[row1_mm][col3] = stoi(val8);
		 	             col3++;
		 	         }
		 row1_mm++;
	 }

	 int row2_mm =0;
	 while (getline(mm_9_file, line9) && row2_mm < dense_2) {
		 istringstream iss9(line9);
		 int col4 = 0;
		 while (getline(iss9, val9, ',')  && col4 < dense_1) { //Within a line, reading values one by one with ',' as delimiter.
		 	        	 mm_weights_9[row2_mm][col4] = stoi(val9);
		 	             col4++;
		 	         }
		 row2_mm++;
	 }

	 int row3_mm =0;
	 while (getline(mm_10_file, line10) && row3_mm < dense_3) {
		 istringstream iss10(line10);
		 int col5 = 0;
		 while (getline(iss10, val10, ',')  && col5 < dense_2) { //Within a line, reading values one by one with ',' as delimiter.
		 	        	 mm_weights_10[row3_mm][col5] = stoi(val10);
		 	        	//cout << "MM_Weight -------" << stoi(val10) << " -- " << col5 << std::endl;
		 	             col5++;
		 	         }
		 row3_mm++;
	 }

	 mm_0_file.close(); mm_1_file.close();mm_2_file.close();mm_3_file.close();
	 mm_4_file.close(); mm_5_file.close();mm_6_file.close();mm_7_file.close();
	 mm_8_file.close(); mm_9_file.close();mm_10_file.close();

	 //----------------------------------------------------------------------
	 //-------------- Loading thresholds for MT operation -------------------
	 ifstream fmt_0,fmt_1,fmt_2,fmt_3,fmt_4,fmt_5,fmt_6,fmt_7,fmt_8,fmt_9,fmt_10,fmt_11,fmt_12,fmt_13,fmt_14,fmt_15,fmt_16,fmt_17,fmt_18,fmt_19,fmt_20;
	 fmt_0.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_0_weights.txt");
	 fmt_1.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_1_weights.txt");
	 fmt_2.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_2_weights.txt");
	 fmt_3.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_3_weights.txt");
	 fmt_4.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_4_weights.txt");
	 fmt_5.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_5_weights.txt");
	 fmt_6.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_6_weights.txt");
	 fmt_7.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_7_weights.txt");
	 fmt_8.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_8_weights.txt");
	 fmt_9.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_9_weights.txt");
	 fmt_10.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_10_weights.txt");
	 fmt_11.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_11_weights.txt");
	 fmt_12.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_12_weights.txt");
	 fmt_13.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_13_weights.txt");
	 fmt_14.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_14_weights.txt");
	 fmt_15.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_15_weights.txt");
	 fmt_16.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_16_weights.txt");
	 fmt_17.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_17_weights.txt");
	 fmt_18.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_18_weights.txt");
	 fmt_19.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_19_weights.txt");
	 fmt_20.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/qlstm_streamlined_model_weights/mt_20_weights.txt");
	 if (!fmt_0.is_open() || !fmt_1.is_open() || !fmt_2.is_open() || !fmt_3.is_open() ||
	 	 !fmt_4.is_open() || !fmt_5.is_open() || !fmt_6.is_open() || !fmt_7.is_open() ||
		 !fmt_8.is_open() || !fmt_9.is_open() || !fmt_10.is_open() || !fmt_11.is_open() ||
		 !fmt_12.is_open() || !fmt_13.is_open() || !fmt_14.is_open() || !fmt_15.is_open() || !fmt_16.is_open()
		 || !fmt_17.is_open()  || !fmt_18.is_open() || !fmt_19.is_open() || !fmt_20.is_open()) {
	 	      std::cerr << "Error opening some file" << std::endl;
	 	      return 1;
	 	}
	 string l0,v0,l1,v1,l2,v2,l3,v3,l4,v4,l5,v5,l6,v6,l7,v7,l8,v8,l9,v9,l10,v10,l11,v11,l12,v12,l13,v13,l14,v14,l15,v15,l16,v16,l17,v17,l18,v18,l19,v19,l20,v20;
	 int row1_mt =0;
	 //4 2D activation matrices updated.
		 while (getline(fmt_3, l3) && getline(fmt_4, l4) && getline(fmt_5, l5) && getline(fmt_6, l6) &&  row1_mt < Out_N) {
			 	 istringstream issmt3(l3),issmt4(l4),issmt5(l5),issmt6(l6);
			 	 int col3 = 0;
		         while (getline(issmt3, v3, ',') && getline(issmt4, v4, ',') && getline(issmt5, v5, ',') && getline(issmt6, v6, ',') && col3 < Act_N_62) { //Within a line, reading values one by one with ',' as delimiter.
		        	 mt_weights_3[row1_mt][col3] = stoi(v3);//stoi() : Converts strings to int
		        	 mt_weights_4[row1_mt][col3] = stoi(v4);
		        	 mt_weights_5[row1_mt][col3] = stoi(v5);
		        	 mt_weights_6[row1_mt][col3] = stoi(v6);
		             col3++;
		         }
		         row1_mt++;
		     }

	// 2 2D activation matrices for the classification part.
		 int row_d1=0;
		 while (getline(fmt_19, l19) &&  row_d1 < dense_1) {
			 	 istringstream issmt19(l19);
			 	 int col_d1 = 0;
		         while (getline(issmt19, v19, ',') && col_d1 < Act_N_255) { //Within a line, reading values one by one with ',' as delimiter.
		        	 mt_weights_19[row_d1][col_d1] = stoi(v19);//stoi() : Converts strings to int
		        	 col_d1++;
		         }
		         row_d1++;
		     }

		 int row_d2=0;
		 while (getline(fmt_20, l20) &&  row_d2 < dense_2) {
			 	 istringstream issmt20(l20);
			 	 int col_d2 = 0;
		         while (getline(issmt20, v20, ',') && col_d2 < Act_N_255) { //Within a line, reading values one by one with ',' as delimiter.
		        	 mt_weights_20[row_d2][col_d2] = stoi(v20);//stoi() : Converts strings to int
		             col_d2++;
		         }
		         row_d2++;
		     }


		 //------ Remaining MT matrices : Dividing into three loops of 255,63 and 62 values.
		 int row2_mt = 0;
		 while (getline(fmt_0, l0) && getline(fmt_1, l1) && getline(fmt_2, l2) && getline(fmt_7, l7) && getline(fmt_8, l8) && getline(fmt_9, l9) && getline(fmt_10, l10) &&
				getline(fmt_11, l11) && getline(fmt_12, l12) && getline(fmt_13, l13) && getline(fmt_14, l14) &&
				getline(fmt_15, l15) && getline(fmt_16, l16) && getline(fmt_17, l17) && getline(fmt_18, l18)   && row2_mt < 1) { // && getline(fmt_17, l17)
		 istringstream issmt0(l0), issmt1(l1),issmt2(l2),issmt7(l7),issmt8(l8),issmt9(l9),issmt10(l10),issmt11(l11),issmt12(l12)
		 ,issmt13(l13),issmt14(l14),issmt15(l15),issmt16(l16),issmt17(l17),issmt18(l18);
		 int col4 = 0,col5 = 0,col6 = 0;
		 //First loop for 255 activation thresholds
         while (getline(issmt0, v0, ',') && getline(issmt1, v1, ',') && getline(issmt16, v16, ',') && getline(issmt17, v17, ',')
        		 && getline(issmt18, v18, ',') && col4 < Act_N_255) { //Within a line, reading values one by one with ',' as delimiter.
		 		mt_weights_0[col4] = stof(v0);//stof() : converts strings to float.
		 		mt_weights_1[col4] = stof(v1);
		 		mt_weights_16[col4] = stoi(v16);
		 		mt_weights_17[col4] = stoi(v17);
		 		mt_weights_18[col4] = stoi(v18);
		 		col4++;
		 }

		 //Second loop for 63 activation thresholds
         while (getline(issmt2, v2, ',') && getline(issmt7, v7, ',') && getline(issmt8, v8, ',') &&
        		 getline(issmt9, v9, ',') && getline(issmt10, v10, ',') && getline(issmt15, v15, ',') && col5 < Act_N_63) { //Within a line, reading values one by one with ',' as delimiter.
        	 	mt_weights_2[col5] = stof(v2);
		 		mt_weights_7[col5] = stoi(v7);
		 		mt_weights_8[col5] = stoi(v8);
		 		mt_weights_9[col5] = stoi(v9);
		 		mt_weights_10[col5] = stoi(v10);
		 		mt_weights_15[col5] = stoi(v15);
		 		//cout << stof(v12)  <<"---" <<col5 << std::endl;
		 		col5++;
		 }

		 //Third loop for 62 activation thresholds
         while (getline(issmt11, v11, ',') && getline(issmt12, v12, ',') && getline(issmt13, v13, ',') &&
        		 getline(issmt14, v14, ',') && col6 < Act_N_62) { //Within a line, reading values one by one with ',' as delimiter.
        	 	mt_weights_11[col6] = stoi(v11);
		 		mt_weights_12[col6] = stoi(v12);
		 		mt_weights_13[col6] = stoi(v13);
		 		mt_weights_14[col6] = stoi(v14);
		 		col6++;
		 }

         	 row2_mt++;
		 }

		 fmt_0.close(),fmt_1.close(),fmt_2.close(),fmt_3.close(),fmt_4.close(),fmt_5.close(),fmt_6.close(),
	     fmt_7.close(),fmt_8.close(),fmt_9.close(),fmt_10.close(),fmt_11.close(),fmt_12.close(),fmt_13.close();
		 fmt_14.close(),fmt_15.close(),fmt_16.close(),fmt_17.close(),fmt_18.close(),fmt_19.close(),fmt_20.close();

		hls::stream<qlstm_f32_t> ft_1;
		hls::stream<qlstm_int8_t> mt_1_test;
		hls::stream<qlstm_uint8_t> mt_2_test;
		hls::stream<qlstm_uint6_t> uint6_test;
		hls::stream<qlstm_int32_t> mm_1_test;
		hls::stream<qlstm_int7_t> mt_3_test;
		hls::stream<qlstm_int6_t> mt_4_test;


	//Module instantiation for testing	
	qlstm_top_2(x_input_final,final_output);

	for(int i=0;i<num_test_inputs;++i){
	ap_axis<32,2,5,6> data_out;
	final_output.read(data_out);
	cout << data_out.data << "   " << i << std::endl;
	}
	return 0;
}