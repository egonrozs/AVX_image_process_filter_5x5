#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include <iostream>
#include "stdio.h"

#include "defs.h"

#define UNROLL 1
//Teljesítményben 1537 1500 in VS 1.0-ig ez a legjobb
void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst)
{
#pragma omp parallel
#pragma omp for schedule(dynamic,1)nowait
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < 3 * imgWidth; W += 32)
		{

			__m256i loaded_value0 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H) * imgWidthF + (W)]);
			__m256i loaded_value1 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H) * imgWidthF + (3 + W)]);
			__m256i loaded_value2 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H) * imgWidthF + (6 + W)]);
			__m256i loaded_value3 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H) * imgWidthF + (9 + W)]);
			__m256i loaded_value4 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H) * imgWidthF + (12 + W)]);

			__m256i loaded_value5 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (W)]);
			__m256i loaded_value6 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (3 + W)]);
			__m256i loaded_value7 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (6 + W)]);
			__m256i loaded_value8 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (9 + W)]);
			__m256i loaded_value9 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (12 + W)]);

			__m256i loaded_value10 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (W)]);
			__m256i loaded_value11 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (3 + W)]);
			__m256i loaded_value12 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (6 + W)]);
			__m256i loaded_value13 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (9 + W)]);
			__m256i loaded_value14 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (12 + W)]);

			__m256i loaded_value15 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (W)]);
			__m256i loaded_value16 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (3 + W)]);
			__m256i loaded_value17 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (6 + W)]);
			__m256i loaded_value18 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (9 + W)]);
			__m256i loaded_value19 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (12 + W)]);

			__m256i loaded_value20 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (W)]);
			__m256i loaded_value21 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (3 + W)]);
			__m256i loaded_value22 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (6 + W)]);
			__m256i loaded_value23 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (9 + W)]);
			__m256i loaded_value24 = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (12 + W)]);

			//Section 1
			__m256i result0 = _mm256_min_epu8(loaded_value0, loaded_value1);
			__m256i result1 = _mm256_max_epu8(loaded_value0, loaded_value1);

			__m256i result2 = _mm256_min_epu8(loaded_value2, loaded_value3);
			__m256i result3 = _mm256_max_epu8(loaded_value2, loaded_value3);

			__m256i result4 = _mm256_min_epu8(loaded_value4, loaded_value5);
			__m256i result5 = _mm256_max_epu8(loaded_value4, loaded_value5);

			__m256i  result6 = _mm256_min_epu8(loaded_value6, loaded_value7);
			__m256i  result7 = _mm256_max_epu8(loaded_value6, loaded_value7);

			__m256i  result8 = _mm256_min_epu8(loaded_value8, loaded_value9);
			__m256i  result9 = _mm256_max_epu8(loaded_value8, loaded_value9);

			__m256i  result10 = _mm256_min_epu8(loaded_value10, loaded_value11);
			__m256i  result11 = _mm256_max_epu8(loaded_value10, loaded_value11);

			__m256i  result12 = _mm256_min_epu8(loaded_value12, loaded_value13);
			__m256i  result13 = _mm256_max_epu8(loaded_value12, loaded_value13);

			__m256i  result14 = _mm256_min_epu8(loaded_value14, loaded_value15);
			__m256i  result15 = _mm256_max_epu8(loaded_value14, loaded_value15);

			__m256i  result16 = _mm256_min_epu8(loaded_value16, loaded_value17);
			__m256i  result17 = _mm256_max_epu8(loaded_value16, loaded_value17);

			__m256i  result18 = _mm256_min_epu8(loaded_value18, loaded_value19);
			__m256i  result19 = _mm256_max_epu8(loaded_value18, loaded_value19);

			__m256i  result20 = _mm256_min_epu8(loaded_value20, loaded_value21);
			__m256i  result21 = _mm256_max_epu8(loaded_value20, loaded_value21);

			__m256i  result22 = _mm256_min_epu8(loaded_value22, loaded_value23);
			__m256i  result23 = _mm256_max_epu8(loaded_value22, loaded_value23);


			//Section 2

			__m256i sec2_result0 = _mm256_min_epu8(result0, result2);
			__m256i sec2_result2 = _mm256_max_epu8(result0, result2);


			__m256i sec2_result1 = _mm256_min_epu8(result1, result3);
			__m256i sec2_result3 = _mm256_max_epu8(result1, result3);


			__m256i sec2_result4 = _mm256_min_epu8(result4, result6);
			__m256i sec2_result6 = _mm256_max_epu8(result4, result6);


			__m256i sec2_result5 = _mm256_min_epu8(result5, result7);
			__m256i sec2_result7 = _mm256_max_epu8(result5, result7);


			__m256i sec2_result8 = _mm256_min_epu8(result8, result10);
			__m256i sec2_result10 = _mm256_max_epu8(result8, result10);


			__m256i sec2_result9 = _mm256_min_epu8(result9, result11);
			__m256i sec2_result11 = _mm256_max_epu8(result9, result11);


			__m256i sec2_result12 = _mm256_min_epu8(result12, result14);
			__m256i sec2_result14 = _mm256_max_epu8(result12, result14);


			__m256i sec2_result13 = _mm256_min_epu8(result13, result15);
			__m256i sec2_result15 = _mm256_max_epu8(result13, result15);


			__m256i sec2_result16 = _mm256_min_epu8(result16, result18);
			__m256i sec2_result18 = _mm256_max_epu8(result16, result18);


			__m256i sec2_result17 = _mm256_min_epu8(result17, result19);
			__m256i sec2_result19 = _mm256_max_epu8(result17, result19);


			__m256i sec2_result20 = _mm256_min_epu8(result20, result22);
			__m256i sec2_result22 = _mm256_max_epu8(result20, result22);


			__m256i sec2_result21 = _mm256_min_epu8(result21, result23);
			__m256i sec2_result23 = _mm256_max_epu8(result21, result23);



			//Section 3
			__m256i sec3_result1 = _mm256_min_epu8(sec2_result1, sec2_result2);
			__m256i sec3_result2 = _mm256_max_epu8(sec2_result1, sec2_result2);


			__m256i sec3_result5 = _mm256_min_epu8(sec2_result5, sec2_result6);
			__m256i sec3_result6 = _mm256_max_epu8(sec2_result5, sec2_result6);


			__m256i sec3_result9 = _mm256_min_epu8(sec2_result9, sec2_result10);
			__m256i sec3_result10 = _mm256_max_epu8(sec2_result9, sec2_result10);


			__m256i sec3_result13 = _mm256_min_epu8(sec2_result13, sec2_result14);
			__m256i sec3_result14 = _mm256_max_epu8(sec2_result13, sec2_result14);


			__m256i sec3_result17 = _mm256_min_epu8(sec2_result17, sec2_result18);
			__m256i sec3_result18 = _mm256_max_epu8(sec2_result17, sec2_result18);


			__m256i sec3_result21 = _mm256_min_epu8(sec2_result21, sec2_result22);
			__m256i sec3_result22 = _mm256_max_epu8(sec2_result21, sec2_result22);

			//Section 4

			__m256i sec4_result0 = _mm256_min_epu8(sec2_result0, sec2_result4);
			__m256i sec4_result4 = _mm256_max_epu8(sec2_result0, sec2_result4);


			__m256i sec4_result1 = _mm256_min_epu8(sec3_result1, sec3_result5);
			__m256i sec4_result5 = _mm256_max_epu8(sec3_result1, sec3_result5);

			__m256i sec4_result2 = _mm256_min_epu8(sec3_result2, sec3_result6);
			__m256i sec4_result6 = _mm256_max_epu8(sec3_result2, sec3_result6);


			__m256i sec4_result3 = _mm256_min_epu8(sec2_result3, sec2_result7);
			__m256i sec4_result7 = _mm256_max_epu8(sec2_result3, sec2_result7);


			__m256i sec4_result8 = _mm256_min_epu8(sec2_result8, sec2_result12);
			__m256i sec4_result12 = _mm256_max_epu8(sec2_result8, sec2_result12);


			__m256i sec4_result9 = _mm256_min_epu8(sec3_result9, sec3_result13);
			__m256i sec4_result13 = _mm256_max_epu8(sec3_result9, sec3_result13);


			__m256i sec4_result10 = _mm256_min_epu8(sec3_result10, sec3_result14);
			__m256i sec4_result14 = _mm256_max_epu8(sec3_result10, sec3_result14);


			__m256i sec4_result11 = _mm256_min_epu8(sec2_result11, sec2_result15);
			__m256i sec4_result15 = _mm256_max_epu8(sec2_result11, sec2_result15);


			__m256i sec4_result16 = _mm256_min_epu8(sec2_result16, sec2_result20);
			__m256i sec4_result20 = _mm256_max_epu8(sec2_result16, sec2_result20);


			__m256i sec4_result17 = _mm256_min_epu8(sec3_result17, sec3_result21);
			__m256i sec4_result21 = _mm256_max_epu8(sec3_result17, sec3_result21);


			__m256i sec4_result18 = _mm256_min_epu8(sec3_result18, sec3_result22);
			__m256i sec4_result22 = _mm256_max_epu8(sec3_result18, sec3_result22);


			__m256i sec4_result19 = _mm256_min_epu8(sec2_result19, sec2_result23);
			__m256i sec4_result23 = _mm256_max_epu8(sec2_result19, sec2_result23);

			//Section 5

			__m256i sec5_result2 = _mm256_min_epu8(sec4_result2, sec4_result4);
			__m256i sec5_result4 = _mm256_max_epu8(sec4_result2, sec4_result4);


			__m256i sec5_result3 = _mm256_min_epu8(sec4_result3, sec4_result5);
			__m256i sec5_result5 = _mm256_max_epu8(sec4_result3, sec4_result5);


			__m256i sec5_result10 = _mm256_min_epu8(sec4_result10, sec4_result12);
			__m256i sec5_result12 = _mm256_max_epu8(sec4_result10, sec4_result12);


			__m256i sec5_result11 = _mm256_min_epu8(sec4_result11, sec4_result13);
			__m256i sec5_result13 = _mm256_max_epu8(sec4_result11, sec4_result13);

			__m256i sec5_result18 = _mm256_min_epu8(sec4_result18, sec4_result20);
			__m256i sec5_result20 = _mm256_max_epu8(sec4_result18, sec4_result20);


			__m256i sec5_result19 = _mm256_min_epu8(sec4_result19, sec4_result21);
			__m256i sec5_result21 = _mm256_max_epu8(sec4_result19, sec4_result21);


			//Section 6

			__m256i sec6_result1 = _mm256_min_epu8(sec4_result1, sec5_result2);
			__m256i sec6_result2 = _mm256_max_epu8(sec4_result1, sec5_result2);


			__m256i sec6_result3 = _mm256_min_epu8(sec5_result3, sec5_result4);
			__m256i sec6_result4 = _mm256_max_epu8(sec5_result3, sec5_result4);


			__m256i sec6_result5 = _mm256_min_epu8(sec5_result5, sec4_result6);
			__m256i sec6_result6 = _mm256_max_epu8(sec5_result5, sec4_result6);


			__m256i sec6_result9 = _mm256_min_epu8(sec4_result9, sec5_result10);
			__m256i sec6_result10 = _mm256_max_epu8(sec4_result9, sec5_result10);


			__m256i sec6_result11 = _mm256_min_epu8(sec5_result11, sec5_result12);
			__m256i sec6_result12 = _mm256_max_epu8(sec5_result11, sec5_result12);


			__m256i sec6_result13 = _mm256_min_epu8(sec5_result13, sec4_result14);
			__m256i sec6_result14 = _mm256_max_epu8(sec5_result13, sec4_result14);


			__m256i sec6_result17 = _mm256_min_epu8(sec4_result17, sec5_result18);
			__m256i sec6_result18 = _mm256_max_epu8(sec4_result17, sec5_result18);


			__m256i sec6_result19 = _mm256_min_epu8(sec5_result19, sec5_result20);
			__m256i sec6_result20 = _mm256_max_epu8(sec5_result19, sec5_result20);


			__m256i sec6_result21 = _mm256_min_epu8(sec5_result21, sec4_result22);
			__m256i sec6_result22 = _mm256_max_epu8(sec5_result21, sec4_result22);


			//Section 7

			__m256i sec7_result0 = _mm256_min_epu8(sec4_result0, sec4_result8);
			__m256i sec7_result8 = _mm256_max_epu8(sec4_result0, sec4_result8);


			__m256i sec7_result1 = _mm256_min_epu8(sec6_result1, sec6_result9);
			__m256i sec7_result9 = _mm256_max_epu8(sec6_result1, sec6_result9);


			__m256i sec7_result2 = _mm256_min_epu8(sec6_result2, sec6_result10);
			__m256i sec7_result10 = _mm256_max_epu8(sec6_result2, sec6_result10);


			__m256i sec7_result3 = _mm256_min_epu8(sec6_result3, sec6_result11);
			__m256i sec7_result11 = _mm256_max_epu8(sec6_result3, sec6_result11);


			__m256i sec7_result4 = _mm256_min_epu8(sec6_result4, sec6_result12);
			__m256i sec7_result12 = _mm256_max_epu8(sec6_result4, sec6_result12);


			__m256i sec7_result5 = _mm256_min_epu8(sec6_result5, sec6_result13);
			__m256i sec7_result13 = _mm256_max_epu8(sec6_result5, sec6_result13);


			__m256i sec7_result6 = _mm256_min_epu8(sec6_result6, sec6_result14);
			__m256i sec7_result14 = _mm256_max_epu8(sec6_result6, sec6_result14);


			__m256i sec7_result7 = _mm256_min_epu8(sec4_result7, sec4_result15);
			__m256i sec7_result15 = _mm256_max_epu8(sec4_result7, sec4_result15);


			__m256i sec7_result16 = _mm256_min_epu8(sec4_result16, loaded_value24);
			__m256i sec7_result24 = _mm256_max_epu8(sec4_result16, loaded_value24);

			//Section 8

			__m256i sec8_result4 = _mm256_min_epu8(sec7_result4, sec7_result8);
			__m256i sec8_result8 = _mm256_max_epu8(sec7_result4, sec7_result8);


			__m256i sec8_result5 = _mm256_min_epu8(sec7_result5, sec7_result9);
			__m256i sec8_result9 = _mm256_max_epu8(sec7_result5, sec7_result9);


			__m256i sec8_result6 = _mm256_min_epu8(sec7_result6, sec7_result10);
			__m256i sec8_result10 = _mm256_max_epu8(sec7_result6, sec7_result10);


			__m256i sec8_result7 = _mm256_min_epu8(sec7_result7, sec7_result11);
			__m256i sec8_result11 = _mm256_max_epu8(sec7_result7, sec7_result11);


			__m256i sec8_result20 = _mm256_min_epu8(sec6_result20, sec7_result24);
			__m256i sec8_result24 = _mm256_max_epu8(sec6_result20, sec7_result24);

			//Section 9

			__m256i sec9_result2 = _mm256_min_epu8(sec7_result2, sec8_result4);
			__m256i sec9_result4 = _mm256_max_epu8(sec7_result2, sec8_result4);


			__m256i sec9_result3 = _mm256_min_epu8(sec7_result3, sec8_result5);
			__m256i sec9_result5 = _mm256_max_epu8(sec7_result3, sec8_result5);


			__m256i sec9_result6 = _mm256_min_epu8(sec8_result6, sec8_result8);
			__m256i sec9_result8 = _mm256_max_epu8(sec8_result6, sec8_result8);


			__m256i sec9_result7 = _mm256_min_epu8(sec8_result7, sec8_result9);
			__m256i sec9_result9 = _mm256_max_epu8(sec8_result7, sec8_result9);


			__m256i sec9_result10 = _mm256_min_epu8(sec8_result10, sec7_result12);
			__m256i sec9_result12 = _mm256_max_epu8(sec8_result10, sec7_result12);


			__m256i sec9_result11 = _mm256_min_epu8(sec8_result11, sec7_result13);
			__m256i sec9_result13 = _mm256_max_epu8(sec8_result11, sec7_result13);


			__m256i sec9_result18 = _mm256_min_epu8(sec6_result18, sec8_result20);
			__m256i sec9_result20 = _mm256_max_epu8(sec6_result18, sec8_result20);


			__m256i sec9_result19 = _mm256_min_epu8(sec6_result19, sec6_result21);
			__m256i sec9_result21 = _mm256_max_epu8(sec6_result19, sec6_result21);


			__m256i sec9_result22 = _mm256_min_epu8(sec6_result22, sec8_result24);
			__m256i sec9_result24 = _mm256_max_epu8(sec6_result22, sec8_result24);

			//Section 10

			__m256i sec10_result1 = _mm256_min_epu8(sec7_result1, sec9_result2);
			__m256i sec10_result2 = _mm256_max_epu8(sec7_result1, sec9_result2);


			__m256i sec10_result3 = _mm256_min_epu8(sec9_result3, sec9_result4);
			__m256i sec10_result4 = _mm256_max_epu8(sec9_result3, sec9_result4);


			__m256i sec10_result5 = _mm256_min_epu8(sec9_result5, sec9_result6);
			__m256i sec10_result6 = _mm256_max_epu8(sec9_result5, sec9_result6);


			__m256i sec10_result7 = _mm256_min_epu8(sec9_result7, sec9_result8);
			__m256i sec10_result8 = _mm256_max_epu8(sec9_result7, sec9_result8);


			__m256i sec10_result9 = _mm256_min_epu8(sec9_result9, sec9_result10);
			__m256i sec10_result10 = _mm256_max_epu8(sec9_result9, sec9_result10);


			__m256i sec10_result11 = _mm256_min_epu8(sec9_result11, sec9_result12);
			__m256i sec10_result12 = _mm256_max_epu8(sec9_result11, sec9_result12);


			__m256i sec10_result13 = _mm256_min_epu8(sec9_result13, sec7_result14);
			__m256i sec10_result14 = _mm256_max_epu8(sec9_result13, sec7_result14);


			__m256i sec10_result17 = _mm256_min_epu8(sec6_result17, sec9_result18);
			__m256i sec10_result18 = _mm256_max_epu8(sec6_result17, sec9_result18);


			__m256i sec10_result19 = _mm256_min_epu8(sec9_result19, sec9_result20);
			__m256i sec10_result20 = _mm256_max_epu8(sec9_result19, sec9_result20);


			__m256i sec10_result21 = _mm256_min_epu8(sec9_result21, sec9_result22);
			__m256i sec10_result22 = _mm256_max_epu8(sec9_result21, sec9_result22);


			__m256i sec10_result23 = _mm256_min_epu8(sec4_result23, sec9_result24);
			__m256i sec10_result24 = _mm256_max_epu8(sec4_result23, sec9_result24);

			//Section 11

			__m256i sec11_result0 = _mm256_min_epu8(sec7_result0, sec7_result16);
			__m256i sec11_result16 = _mm256_max_epu8(sec7_result0, sec7_result16);


			__m256i sec11_result1 = _mm256_min_epu8(sec10_result1, sec10_result17);
			__m256i sec11_result17 = _mm256_max_epu8(sec10_result1, sec10_result17);


			__m256i sec11_result2 = _mm256_min_epu8(sec10_result2, sec10_result18);
			__m256i sec11_result18 = _mm256_max_epu8(sec10_result2, sec10_result18);


			__m256i sec11_result3 = _mm256_min_epu8(sec10_result3, sec10_result19);
			__m256i sec11_result19 = _mm256_max_epu8(sec10_result3, sec10_result19);


			__m256i sec11_result4 = _mm256_min_epu8(sec10_result4, sec10_result20);
			__m256i sec11_result20 = _mm256_max_epu8(sec10_result4, sec10_result20);


			__m256i sec11_result5 = _mm256_min_epu8(sec10_result5, sec10_result21);
			__m256i sec11_result21 = _mm256_max_epu8(sec10_result5, sec10_result21);


			__m256i sec11_result6 = _mm256_min_epu8(sec10_result6, sec10_result22);
			__m256i sec11_result22 = _mm256_max_epu8(sec10_result6, sec10_result22);


			__m256i sec11_result7 = _mm256_min_epu8(sec10_result7, sec10_result23);
			__m256i sec11_result23 = _mm256_max_epu8(sec10_result7, sec10_result23);


			__m256i sec11_result8 = _mm256_min_epu8(sec10_result8, sec10_result24);
			__m256i sec11_result24 = _mm256_max_epu8(sec10_result8, sec10_result24);

			//Section 12

			__m256i sec12_result8 = _mm256_min_epu8(sec11_result8, sec11_result16);
			__m256i sec12_result16 = _mm256_max_epu8(sec11_result8, sec11_result16);


			__m256i sec12_result9 = _mm256_min_epu8(sec10_result9, sec11_result17);
			__m256i sec12_result17 = _mm256_max_epu8(sec10_result9, sec11_result17);


			__m256i sec12_result10 = _mm256_min_epu8(sec10_result10, sec11_result18);
			__m256i sec12_result18 = _mm256_max_epu8(sec10_result10, sec11_result18);


			__m256i sec12_result11 = _mm256_min_epu8(sec10_result11, sec11_result19);
			__m256i sec12_result19 = _mm256_max_epu8(sec10_result11, sec11_result19);


			__m256i sec12_result12 = _mm256_min_epu8(sec10_result12, sec11_result20);
			__m256i sec12_result20 = _mm256_max_epu8(sec10_result12, sec11_result20);


			__m256i sec12_result13 = _mm256_min_epu8(sec10_result13, sec11_result21);
			__m256i sec12_result21 = _mm256_max_epu8(sec10_result13, sec11_result21);

			//Section 13

			__m256i sec13_result6 = _mm256_min_epu8(sec11_result6, sec12_result10);
			__m256i sec13_result10 = _mm256_max_epu8(sec11_result6, sec12_result10);


			__m256i sec13_result7 = _mm256_min_epu8(sec11_result7, sec12_result11);
			__m256i sec13_result11 = _mm256_max_epu8(sec11_result7, sec12_result11);


			__m256i sec13_result12 = _mm256_min_epu8(sec12_result12, sec12_result16);
			__m256i sec13_result16 = _mm256_max_epu8(sec12_result12, sec12_result16);


			__m256i sec13_result13 = _mm256_min_epu8(sec12_result13, sec12_result17);
			__m256i sec13_result17 = _mm256_max_epu8(sec12_result13, sec12_result17);

			//Section 14

			__m256i sec14_result10 = _mm256_min_epu8(sec13_result10, sec13_result12);
			__m256i sec14_result12 = _mm256_max_epu8(sec13_result10, sec13_result12);


			__m256i sec14_result11 = _mm256_min_epu8(sec13_result11, sec13_result13);
			__m256i sec14_result13 = _mm256_max_epu8(sec13_result11, sec13_result13);

			//Last Section
			result12 = _mm256_max_epu8(sec14_result11, sec14_result12);

			int pixel_dst = H * imgWidth * 3 + W;
			_mm256_storeu_epi8(imgDst + pixel_dst, result12);
		};
	};
}
//ez az unroll ad csak jó eredményt a többi belecsúszott egy hiba