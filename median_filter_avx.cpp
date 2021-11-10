#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include <iostream>
#include "stdio.h"

#include "defs.h"

#define UNROLL 1
//V1.0 BMS2 
void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst)
{


#pragma omp parallel
#pragma omp for schedule(dynamic,1)nowait
	for (int H = 0; H < imgHeight; H += 2)
	{
		for (int W = 0; W < 3 * imgWidth; W += 32)
		{

			__m256i loaded_value0 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H)*imgWidthF + (W)]);
			__m256i loaded_value1 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H)*imgWidthF + (3 + W)]);
			__m256i loaded_value2 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H)*imgWidthF + (6 + W)]);
			__m256i loaded_value3 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H)*imgWidthF + (9 + W)]);
			__m256i loaded_value4 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H)*imgWidthF + (12 + W)]);

			__m256i loaded_value5 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (W)]);
			__m256i loaded_value6 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (3 + W)]);
			__m256i loaded_value7 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (6 + W)]);
			__m256i loaded_value8 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (9 + W)]);
			__m256i loaded_value9 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 1) * imgWidthF + (12 + W)]);

			__m256i loaded_value10 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (W)]);
			__m256i loaded_value11 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (3 + W)]);
			__m256i loaded_value12 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (6 + W)]);
			__m256i loaded_value13 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (9 + W)]);
			__m256i loaded_value14 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (12 + W)]);

			__m256i loaded_value15 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (W)]);
			__m256i loaded_value16 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (3 + W)]);
			__m256i loaded_value17 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (6 + W)]);
			__m256i loaded_value18 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (9 + W)]);
			__m256i loaded_value19 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (12 + W)]);

			__m256i loaded_value20 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (W)]);
			__m256i loaded_value21 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (3 + W)]);
			__m256i loaded_value22 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (6 + W)]);
			__m256i loaded_value23 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (9 + W)]);
			__m256i loaded_value24 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 4) * imgWidthF + (12 + W)]);

			__m256i loaded_value25 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 5) * imgWidthF + (W)]);
			__m256i loaded_value26 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 5) * imgWidthF + (3 + W)]);
			__m256i loaded_value27 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 5) * imgWidthF + (6 + W)]);
			__m256i loaded_value28 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 5) * imgWidthF + (9 + W)]);
			__m256i loaded_value29 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 5) * imgWidthF + (12 + W)]);


			//Közös halmaz szûrése
			//Section 1
			__m256i result5 = _mm256_min_epu8(loaded_value5, loaded_value6);
			__m256i result6 = _mm256_max_epu8(loaded_value5, loaded_value6);

			__m256i result7 = _mm256_min_epu8(loaded_value7, loaded_value8);
			__m256i result8 = _mm256_max_epu8(loaded_value7, loaded_value8);

			__m256i result9 = _mm256_min_epu8(loaded_value9, loaded_value10);
			__m256i result10 = _mm256_max_epu8(loaded_value9, loaded_value10);

			__m256i result11 = _mm256_min_epu8(loaded_value11, loaded_value12);
			__m256i result12 = _mm256_max_epu8(loaded_value11, loaded_value12);

			__m256i result13 = _mm256_min_epu8(loaded_value13, loaded_value14);
			__m256i result14 = _mm256_max_epu8(loaded_value13, loaded_value14);

			__m256i result15 = _mm256_min_epu8(loaded_value15, loaded_value16);
			__m256i result16 = _mm256_max_epu8(loaded_value15, loaded_value16);

			__m256i result17 = _mm256_min_epu8(loaded_value17, loaded_value18);
			__m256i result18 = _mm256_max_epu8(loaded_value17, loaded_value18);

			__m256i result19 = _mm256_min_epu8(loaded_value19, loaded_value20);
			__m256i result20 = _mm256_max_epu8(loaded_value19, loaded_value20);

			__m256i result21 = _mm256_min_epu8(loaded_value21, loaded_value22);
			__m256i result22 = _mm256_max_epu8(loaded_value21, loaded_value22);

			__m256i result23 = _mm256_min_epu8(loaded_value23, loaded_value24);
			__m256i result24 = _mm256_max_epu8(loaded_value23, loaded_value24);




			//Section 2
			__m256i tmp = _mm256_max_epu8(result5, result7);
			result5 = _mm256_min_epu8(result5, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result6, result8);
			result6 = _mm256_min_epu8(result6, result8);
			result8 = tmp;

			tmp = _mm256_max_epu8(result9, result11);
			result9 = _mm256_min_epu8(result9, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result10, result12);
			result10 = _mm256_min_epu8(result10, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result13, result15);
			result13 = _mm256_min_epu8(result13, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result14, result16);
			result14 = _mm256_min_epu8(result14, result16);
			result16 = tmp;

			tmp = _mm256_max_epu8(result17, result19);
			result17 = _mm256_min_epu8(result17, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result18, result20);
			result18 = _mm256_min_epu8(result18, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result21, result23);
			result21 = _mm256_min_epu8(result21, result23);
			result23 = tmp;

			tmp = _mm256_max_epu8(result22, result24);
			result22 = _mm256_min_epu8(result22, result24);
			result24 = tmp;

			//Section 3

			tmp = _mm256_max_epu8(result6, result7);
			result6 = _mm256_min_epu8(result6, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result10, result11);
			result10 = _mm256_min_epu8(result10, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result14, result15);
			result14 = _mm256_min_epu8(result14, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result18, result19);
			result18 = _mm256_min_epu8(result18, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result22, result23);
			result22 = _mm256_min_epu8(result22, result23);
			result23 = tmp;

			//Section 4

			tmp = _mm256_max_epu8(result5, result9);
			result5 = _mm256_min_epu8(result5, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result6, result10);
			result6 = _mm256_min_epu8(result6, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result7, result11);
			result7 = _mm256_min_epu8(result7, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result8, result12);
			result8 = _mm256_min_epu8(result8, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result13, result17);
			result13 = _mm256_min_epu8(result13, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result14, result18);
			result14 = _mm256_min_epu8(result14, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result15, result19);
			result15 = _mm256_min_epu8(result15, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result16, result20);
			result16 = _mm256_min_epu8(result16, result20);
			result20 = tmp;

			//Section 5

			tmp = _mm256_max_epu8(result7, result9);
			result7 = _mm256_min_epu8(result7, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result8, result10);
			result8 = _mm256_min_epu8(result8, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result15, result17);
			result15 = _mm256_min_epu8(result15, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result16, result18);
			result16 = _mm256_min_epu8(result16, result18);
			result18 = tmp;

			//Section 6

			tmp = _mm256_max_epu8(result6, result7);
			result6 = _mm256_min_epu8(result6, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result8, result9);
			result8 = _mm256_min_epu8(result8, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result10, result11);
			result10 = _mm256_min_epu8(result10, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result14, result15);
			result14 = _mm256_min_epu8(result14, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result16, result17);
			result16 = _mm256_min_epu8(result16, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result18, result19);
			result18 = _mm256_min_epu8(result18, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result22, result23);
			result22 = _mm256_min_epu8(result22, result23);
			result23 = tmp;

			//Section 7

			tmp = _mm256_max_epu8(result5, result13);
			result5 = _mm256_min_epu8(result5, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result6, result14);
			result6 = _mm256_min_epu8(result6, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result7, result15);
			result7 = _mm256_min_epu8(result7, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result8, result16);
			result8 = _mm256_min_epu8(result8, result16);
			result16 = tmp;

			tmp = _mm256_max_epu8(result9, result17);
			result9 = _mm256_min_epu8(result9, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result10, result18);
			result10 = _mm256_min_epu8(result10, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result11, result19);
			result11 = _mm256_min_epu8(result11, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result12, result20);
			result12 = _mm256_min_epu8(result12, result20);
			result20 = tmp;

			//Section 8

			tmp = _mm256_max_epu8(result9, result13);
			result9 = _mm256_min_epu8(result9, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result10, result14);
			result10 = _mm256_min_epu8(result10, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result11, result15);
			result11 = _mm256_min_epu8(result11, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result12, result16);
			result12 = _mm256_min_epu8(result12, result16);
			result16 = tmp;

			//Section 9

			tmp = _mm256_max_epu8(result7, result9);
			result7 = _mm256_min_epu8(result7, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result8, result10);
			result8 = _mm256_min_epu8(result8, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result11, result13);
			result11 = _mm256_min_epu8(result11, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result12, result14);
			result12 = _mm256_min_epu8(result12, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result15, result17);
			result15 = _mm256_min_epu8(result15, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result16, result18);
			result16 = _mm256_min_epu8(result16, result18);
			result18 = tmp;

			//Section 10

			tmp = _mm256_max_epu8(result6, result7);
			result6 = _mm256_min_epu8(result6, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result8, result9);
			result8 = _mm256_min_epu8(result8, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result10, result11);
			result10 = _mm256_min_epu8(result10, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result12, result13);
			result12 = _mm256_min_epu8(result12, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result14, result15);
			result14 = _mm256_min_epu8(result14, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result16, result17);
			result16 = _mm256_min_epu8(result16, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result18, result19);
			result18 = _mm256_min_epu8(result18, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result22, result23);
			result22 = _mm256_min_epu8(result22, result23);
			result23 = tmp;

			//Section 11

			tmp = _mm256_max_epu8(result5, result21);
			result5 = _mm256_min_epu8(result5, result21);
			result21 = tmp;

			tmp = _mm256_max_epu8(result6, result22);
			result6 = _mm256_min_epu8(result6, result22);
			result22 = tmp;

			tmp = _mm256_max_epu8(result7, result23);
			result7 = _mm256_min_epu8(result7, result23);
			result23 = tmp;

			tmp = _mm256_max_epu8(result8, result24);
			result8 = _mm256_min_epu8(result8, result24);
			result24 = tmp;

			//Section 12

			tmp = _mm256_max_epu8(result13, result21);
			result13 = _mm256_min_epu8(result13, result21);
			result21 = tmp;

			tmp = _mm256_max_epu8(result14, result22);
			result14 = _mm256_min_epu8(result14, result22);
			result22 = tmp;

			tmp = _mm256_max_epu8(result15, result23);
			result15 = _mm256_min_epu8(result15, result23);
			result23 = tmp;

			tmp = _mm256_max_epu8(result16, result24);
			result16 = _mm256_min_epu8(result16, result24);
			result24 = tmp;

			//Section 13

			tmp = _mm256_max_epu8(result9, result13);
			result9 = _mm256_min_epu8(result9, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result10, result14);
			result10 = _mm256_min_epu8(result10, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result11, result15);
			result11 = _mm256_min_epu8(result11, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result12, result16);
			result12 = _mm256_min_epu8(result12, result16);
			result16 = tmp;

			tmp = _mm256_max_epu8(result17, result21);
			result17 = _mm256_min_epu8(result17, result21);
			result21 = tmp;

			tmp = _mm256_max_epu8(result18, result22);
			result18 = _mm256_min_epu8(result18, result22);
			result22 = tmp;

			//Section 14

			tmp = _mm256_max_epu8(result11, result13);
			result11 = _mm256_min_epu8(result11, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result12, result14);
			result12 = _mm256_min_epu8(result12, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result15, result17);
			result15 = _mm256_min_epu8(result15, result17);
			result17 = tmp;

			tmp = _mm256_max_epu8(result16, result18);
			result16 = _mm256_min_epu8(result16, result18);
			result18 = tmp;

			//Section 15
			tmp = _mm256_max_epu8(result12, result13);
			result12 = _mm256_min_epu8(result12, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result14, result15);
			result14 = _mm256_min_epu8(result14, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result16, result17);
			result16 = _mm256_min_epu8(result16, result17);
			result17 = tmp;

			//Különbözõ
			// Kernel 1
			//Section 1  
			__m256i kernel1_0 = _mm256_min_epu8(loaded_value0, loaded_value1);
			__m256i kernel1_1 = _mm256_max_epu8(loaded_value0, loaded_value1);

			__m256i kernel1_2 = _mm256_min_epu8(loaded_value2, loaded_value3);
			__m256i kernel1_3 = _mm256_max_epu8(loaded_value2, loaded_value3);

			__m256i kernel1_4 = _mm256_min_epu8(loaded_value4, result12);
			__m256i kernel1_5 = _mm256_max_epu8(loaded_value4, result12);

			__m256i kernel1_6 = result13;
			__m256i kernel1_7 = result14;

			__m256i kernel1_8 = result15;
			__m256i kernel1_9 = result16;

			//Section 2

			tmp = _mm256_max_epu8(kernel1_0, kernel1_2);
			kernel1_0 = _mm256_min_epu8(kernel1_0, kernel1_2);
			kernel1_2 = tmp;

			tmp = _mm256_max_epu8(kernel1_1, kernel1_3);
			kernel1_1 = _mm256_min_epu8(kernel1_1, kernel1_3);
			kernel1_3 = tmp;

			tmp = _mm256_max_epu8(kernel1_4, kernel1_6);
			kernel1_4 = _mm256_min_epu8(kernel1_4, kernel1_6);
			kernel1_6 = tmp;

			tmp = _mm256_max_epu8(kernel1_5, kernel1_7);
			kernel1_5 = _mm256_min_epu8(kernel1_5, kernel1_7);
			kernel1_7 = tmp;

			tmp = _mm256_max_epu8(kernel1_8, result17);
			kernel1_8 = _mm256_min_epu8(kernel1_8, result17);
			__m256i kernel1_10 = tmp;

			//Section 3

			tmp = _mm256_max_epu8(kernel1_1, kernel1_2);
			kernel1_1 = _mm256_min_epu8(kernel1_1, kernel1_2);
			kernel1_2 = tmp;

			tmp = _mm256_max_epu8(kernel1_5, kernel1_6);
			kernel1_5 = _mm256_min_epu8(kernel1_5, kernel1_6);
			kernel1_6 = tmp;

			tmp = _mm256_max_epu8(kernel1_9, kernel1_10);
			kernel1_9 = _mm256_min_epu8(kernel1_9, kernel1_10);
			kernel1_10 = tmp;

			//Section 4

			tmp = _mm256_max_epu8(kernel1_0, kernel1_4);
			kernel1_0 = _mm256_min_epu8(kernel1_0, kernel1_4);
			kernel1_4 = tmp;

			tmp = _mm256_max_epu8(kernel1_1, kernel1_5);
			kernel1_1 = _mm256_min_epu8(kernel1_1, kernel1_5);
			kernel1_5 = tmp;

			tmp = _mm256_max_epu8(kernel1_2, kernel1_6);
			kernel1_2 = _mm256_min_epu8(kernel1_2, kernel1_6);
			kernel1_6 = tmp;

			tmp = _mm256_max_epu8(kernel1_3, kernel1_7);
			kernel1_3 = _mm256_min_epu8(kernel1_3, kernel1_7);
			kernel1_7 = tmp;

			//Section 5

			tmp = _mm256_max_epu8(kernel1_2, kernel1_4);
			kernel1_2 = _mm256_min_epu8(kernel1_2, kernel1_4);
			kernel1_4 = tmp;

			tmp = _mm256_max_epu8(kernel1_3, kernel1_5);
			kernel1_3 = _mm256_min_epu8(kernel1_3, kernel1_5);
			kernel1_5 = tmp;

			//Section 6

			tmp = _mm256_max_epu8(kernel1_1, kernel1_2);
			kernel1_1 = _mm256_min_epu8(kernel1_1, kernel1_2);
			kernel1_2 = tmp;

			tmp = _mm256_max_epu8(kernel1_3, kernel1_4);
			kernel1_3 = _mm256_min_epu8(kernel1_3, kernel1_4);
			kernel1_4 = tmp;

			tmp = _mm256_max_epu8(kernel1_5, kernel1_6);
			kernel1_5 = _mm256_min_epu8(kernel1_5, kernel1_6);
			kernel1_6 = tmp;

			tmp = _mm256_max_epu8(kernel1_9, kernel1_10);
			kernel1_9 = _mm256_min_epu8(kernel1_9, kernel1_10);
			kernel1_10 = tmp;

			//Section 7

			tmp = _mm256_max_epu8(kernel1_0, kernel1_8);
			kernel1_0 = _mm256_min_epu8(kernel1_0, kernel1_8);
			kernel1_8 = tmp;

			tmp = _mm256_max_epu8(kernel1_1, kernel1_9);
			kernel1_1 = _mm256_min_epu8(kernel1_1, kernel1_9);
			kernel1_9 = tmp;

			tmp = _mm256_max_epu8(kernel1_2, kernel1_10);
			kernel1_2 = _mm256_min_epu8(kernel1_2, kernel1_10);
			kernel1_10 = tmp;

			//Section 8

			tmp = _mm256_max_epu8(kernel1_4, kernel1_8);
			kernel1_4 = _mm256_min_epu8(kernel1_4, kernel1_8);
			kernel1_8 = tmp;

			tmp = _mm256_max_epu8(kernel1_5, kernel1_9);
			kernel1_5 = _mm256_min_epu8(kernel1_5, kernel1_9);
			kernel1_9 = tmp;

			tmp = _mm256_max_epu8(kernel1_6, kernel1_10);
			kernel1_6 = _mm256_min_epu8(kernel1_6, kernel1_10);
			kernel1_10 = tmp;

			//Section 9

			tmp = _mm256_max_epu8(kernel1_3, kernel1_5);
			kernel1_3 = _mm256_min_epu8(kernel1_3, kernel1_5);
			kernel1_5 = tmp;

			tmp = _mm256_max_epu8(kernel1_6, kernel1_8);
			kernel1_6 = _mm256_min_epu8(kernel1_6, kernel1_8);
			kernel1_8 = tmp;

			//Section 10

			tmp = _mm256_max_epu8(kernel1_5, kernel1_6);
			kernel1_5 = _mm256_min_epu8(kernel1_5, kernel1_6);
			kernel1_6 = tmp;

			// Kernel 2
			//Section 1   
			__m256i kernel2_0 = _mm256_min_epu8(loaded_value25, loaded_value26);
			__m256i kernel2_1 = _mm256_max_epu8(loaded_value25, loaded_value26);

			__m256i kernel2_2 = _mm256_min_epu8(loaded_value27, loaded_value28);
			__m256i kernel2_3 = _mm256_max_epu8(loaded_value27, loaded_value28);

			__m256i kernel2_4 = _mm256_min_epu8(loaded_value29, result12);
			__m256i kernel2_5 = _mm256_max_epu8(loaded_value29, result12);

			__m256i kernel2_6 = result13;
			__m256i kernel2_7 = result14;

			__m256i kernel2_8 = result15;
			__m256i kernel2_9 = result16;

			//Section 2

			tmp = _mm256_max_epu8(kernel2_0, kernel2_2);
			kernel2_0 = _mm256_min_epu8(kernel2_0, kernel2_2);
			kernel2_2 = tmp;

			tmp = _mm256_max_epu8(kernel2_1, kernel2_3);
			kernel2_1 = _mm256_min_epu8(kernel2_1, kernel2_3);
			kernel2_3 = tmp;

			tmp = _mm256_max_epu8(kernel2_4, kernel2_6);
			kernel2_4 = _mm256_min_epu8(kernel2_4, kernel2_6);
			kernel2_6 = tmp;

			tmp = _mm256_max_epu8(kernel2_5, kernel2_7);
			kernel2_5 = _mm256_min_epu8(kernel2_5, kernel2_7);
			kernel2_7 = tmp;

			tmp = _mm256_max_epu8(kernel2_8, result17);
			kernel2_8 = _mm256_min_epu8(kernel2_8, result17);
			__m256i kernel2_10 = tmp;

			//Section 3

			tmp = _mm256_max_epu8(kernel2_1, kernel2_2);
			kernel2_1 = _mm256_min_epu8(kernel2_1, kernel2_2);
			kernel2_2 = tmp;

			tmp = _mm256_max_epu8(kernel2_5, kernel2_6);
			kernel2_5 = _mm256_min_epu8(kernel2_5, kernel2_6);
			kernel2_6 = tmp;

			tmp = _mm256_max_epu8(kernel2_9, kernel2_10);
			kernel2_9 = _mm256_min_epu8(kernel2_9, kernel2_10);
			kernel2_10 = tmp;

			//Section 4

			tmp = _mm256_max_epu8(kernel2_0, kernel2_4);
			kernel2_0 = _mm256_min_epu8(kernel2_0, kernel2_4);
			kernel2_4 = tmp;

			tmp = _mm256_max_epu8(kernel2_1, kernel2_5);
			kernel2_1 = _mm256_min_epu8(kernel2_1, kernel2_5);
			kernel2_5 = tmp;

			tmp = _mm256_max_epu8(kernel2_2, kernel2_6);
			kernel2_2 = _mm256_min_epu8(kernel2_2, kernel2_6);
			kernel2_6 = tmp;

			tmp = _mm256_max_epu8(kernel2_3, kernel2_7);
			kernel2_3 = _mm256_min_epu8(kernel2_3, kernel2_7);
			kernel2_7 = tmp;

			//Section 5

			tmp = _mm256_max_epu8(kernel2_2, kernel2_4);
			kernel2_2 = _mm256_min_epu8(kernel2_2, kernel2_4);
			kernel2_4 = tmp;

			tmp = _mm256_max_epu8(kernel2_3, kernel2_5);
			kernel2_3 = _mm256_min_epu8(kernel2_3, kernel2_5);
			kernel2_5 = tmp;

			//Section 6

			tmp = _mm256_max_epu8(kernel2_1, kernel2_2);
			kernel2_1 = _mm256_min_epu8(kernel2_1, kernel2_2);
			kernel2_2 = tmp;

			tmp = _mm256_max_epu8(kernel2_3, kernel2_4);
			kernel2_3 = _mm256_min_epu8(kernel2_3, kernel2_4);
			kernel2_4 = tmp;

			tmp = _mm256_max_epu8(kernel2_5, kernel2_6);
			kernel2_5 = _mm256_min_epu8(kernel2_5, kernel2_6);
			kernel2_6 = tmp;

			tmp = _mm256_max_epu8(kernel2_9, kernel2_10);
			kernel2_9 = _mm256_min_epu8(kernel2_9, kernel2_10);
			kernel2_10 = tmp;

			//Section 7

			tmp = _mm256_max_epu8(kernel2_0, kernel2_8);
			kernel2_0 = _mm256_min_epu8(kernel2_0, kernel2_8);
			kernel2_8 = tmp;

			tmp = _mm256_max_epu8(kernel2_1, kernel2_9);
			kernel2_1 = _mm256_min_epu8(kernel2_1, kernel2_9);
			kernel2_9 = tmp;

			tmp = _mm256_max_epu8(kernel2_2, kernel2_10);
			kernel2_2 = _mm256_min_epu8(kernel2_2, kernel2_10);
			kernel2_10 = tmp;

			//Section 8

			tmp = _mm256_max_epu8(kernel2_4, kernel2_8);
			kernel2_4 = _mm256_min_epu8(kernel2_4, kernel2_8);
			kernel2_8 = tmp;

			tmp = _mm256_max_epu8(kernel2_5, kernel2_9);
			kernel2_5 = _mm256_min_epu8(kernel2_5, kernel2_9);
			kernel2_9 = tmp;

			tmp = _mm256_max_epu8(kernel2_6, kernel2_10);
			kernel2_6 = _mm256_min_epu8(kernel2_6, kernel2_10);
			kernel2_10 = tmp;

			//Section 9

			tmp = _mm256_max_epu8(kernel2_3, kernel2_5);
			kernel2_3 = _mm256_min_epu8(kernel2_3, kernel2_5);
			kernel2_5 = tmp;

			tmp = _mm256_max_epu8(kernel2_6, kernel2_8);
			kernel2_6 = _mm256_min_epu8(kernel2_6, kernel2_8);
			kernel2_8 = tmp;

			//Section 10

			tmp = _mm256_max_epu8(kernel2_5, kernel2_6);
			kernel2_5 = _mm256_min_epu8(kernel2_5, kernel2_6);
			kernel2_6 = tmp;

			int pixel_dst1 = H * imgWidth * 3 + W;
			int pixel_dst2 = (H + 1) * imgWidth * 3 + W;

			_mm256_storeu_si256((__m256i*)(imgDst + pixel_dst1), kernel1_5);
			//
			_mm256_storeu_si256((__m256i*)(imgDst + pixel_dst2), kernel2_5);
		};
	};
}