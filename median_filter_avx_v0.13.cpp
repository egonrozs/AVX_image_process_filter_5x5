#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include <iostream>
#include "stdio.h"

#include "defs.h"

#define UNROLL 1
//V0.13
//V0.13 csak ami kell azt mentem
void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst)
{


#pragma omp parallel
#pragma omp for schedule(dynamic,1)nowait
	for (int H = 0; H < imgHeight; H++)
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

			//Section 1
			__m256i result0 = _mm256_min_epu8(loaded_value0, loaded_value1);
			__m256i result1 = _mm256_max_epu8(loaded_value0, loaded_value1);

			__m256i result2 = _mm256_min_epu8(loaded_value2, loaded_value3);
			__m256i result3 = _mm256_max_epu8(loaded_value2, loaded_value3);

			__m256i result4 = _mm256_min_epu8(loaded_value4, loaded_value5);
			__m256i result5 = _mm256_max_epu8(loaded_value4, loaded_value5);

			__m256i result6 = _mm256_min_epu8(loaded_value6, loaded_value7);
			__m256i result7 = _mm256_max_epu8(loaded_value6, loaded_value7);

			__m256i result8 = _mm256_min_epu8(loaded_value8, loaded_value9);
			__m256i result9 = _mm256_max_epu8(loaded_value8, loaded_value9);

			__m256i result10 = _mm256_min_epu8(loaded_value10, loaded_value11);
			__m256i result11 = _mm256_max_epu8(loaded_value10, loaded_value11);

			__m256i result12 = _mm256_min_epu8(loaded_value12, loaded_value13);
			__m256i result13 = _mm256_max_epu8(loaded_value12, loaded_value13);

			__m256i result14 = _mm256_min_epu8(loaded_value14, loaded_value15);
			__m256i result15 = _mm256_max_epu8(loaded_value14, loaded_value15);

			__m256i result16 = _mm256_min_epu8(loaded_value16, loaded_value17);
			__m256i result17 = _mm256_max_epu8(loaded_value16, loaded_value17);

			__m256i result18 = _mm256_min_epu8(loaded_value18, loaded_value19);
			__m256i result19 = _mm256_max_epu8(loaded_value18, loaded_value19);

			__m256i result20 = _mm256_min_epu8(loaded_value20, loaded_value21);
			__m256i result21 = _mm256_max_epu8(loaded_value20, loaded_value21);

			__m256i result22 = _mm256_min_epu8(loaded_value22, loaded_value23);
			__m256i result23 = _mm256_max_epu8(loaded_value22, loaded_value23);


			//Section 2
			__m256i tmp = _mm256_max_epu8(result0, result2);
			result0 = _mm256_min_epu8(result0, result2);
			result2 = tmp;

			tmp = _mm256_max_epu8(result1, result3);
			result1 = _mm256_min_epu8(result1, result3);
			result3 = tmp;

			tmp = _mm256_max_epu8(result4, result6);
			result4 = _mm256_min_epu8(result4, result6);
			result6 = tmp;

			tmp = _mm256_max_epu8(result5, result7);
			result5 = _mm256_min_epu8(result5, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result8, result10);
			result8 = _mm256_min_epu8(result8, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result9, result11);
			result9 = _mm256_min_epu8(result9, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result12, result14);
			result12 = _mm256_min_epu8(result12, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result13, result15);
			result13 = _mm256_min_epu8(result13, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result16, result18);
			result16 = _mm256_min_epu8(result16, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result17, result19);
			result17 = _mm256_min_epu8(result17, result19);
			result19 = tmp;

			tmp = _mm256_max_epu8(result20, result22);
			result20 = _mm256_min_epu8(result20, result22);
			result22 = tmp;

			tmp = _mm256_max_epu8(result21, result23);
			result21 = _mm256_min_epu8(result21, result23);
			result23 = tmp;



			//Section 3
			tmp = _mm256_max_epu8(result1, result2);
			result1 = _mm256_min_epu8(result1, result2);
			result2 = tmp;

			tmp = _mm256_max_epu8(result5, result6);
			result5 = _mm256_min_epu8(result5, result6);
			result6 = tmp;

			tmp = _mm256_max_epu8(result9, result10);
			result9 = _mm256_min_epu8(result9, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result13, result14);
			result13 = _mm256_min_epu8(result13, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result17, result18);
			result17 = _mm256_min_epu8(result17, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result21, result22);
			result21 = _mm256_min_epu8(result21, result22);
			result22 = tmp;

			//Section 4
			tmp = _mm256_max_epu8(result0, result4);
			result0 = _mm256_min_epu8(result0, result4);
			result4 = tmp;

			tmp = _mm256_max_epu8(result1, result5);
			result1 = _mm256_min_epu8(result1, result5);
			result5 = tmp;

			tmp = _mm256_max_epu8(result2, result6);
			result2 = _mm256_min_epu8(result2, result6);
			result6 = tmp;

			tmp = _mm256_max_epu8(result3, result7);
			result3 = _mm256_min_epu8(result3, result7);
			result7 = tmp;

			tmp = _mm256_max_epu8(result8, result12);
			result8 = _mm256_min_epu8(result8, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result9, result13);
			result9 = _mm256_min_epu8(result9, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result10, result14);
			result10 = _mm256_min_epu8(result10, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result11, result15);
			result11 = _mm256_min_epu8(result11, result15);
			result15 = tmp;

			tmp = _mm256_max_epu8(result16, result20);
			result16 = _mm256_min_epu8(result16, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result17, result21);
			result17 = _mm256_min_epu8(result17, result21);
			result21 = tmp;

			tmp = _mm256_max_epu8(result18, result22);
			result18 = _mm256_min_epu8(result18, result22);
			result22 = tmp;

			tmp = _mm256_max_epu8(result19, result23);
			result19 = _mm256_min_epu8(result19, result23);
			result23 = tmp;

			//Section 5
			tmp = _mm256_max_epu8(result2, result4);
			result2 = _mm256_min_epu8(result2, result4);
			result4 = tmp;

			tmp = _mm256_max_epu8(result3, result5);
			result3 = _mm256_min_epu8(result3, result5);
			result5 = tmp;

			tmp = _mm256_max_epu8(result10, result12);
			result10 = _mm256_min_epu8(result10, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result11, result13);
			result11 = _mm256_min_epu8(result11, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result18, result20);
			result18 = _mm256_min_epu8(result18, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result19, result21);
			result19 = _mm256_min_epu8(result19, result21);
			result21 = tmp;


			//Section 6
			tmp = _mm256_max_epu8(result1, result2);
			result1 = _mm256_min_epu8(result1, result2);
			result2 = tmp;

			tmp = _mm256_max_epu8(result3, result4);
			result3 = _mm256_min_epu8(result3, result4);
			result4 = tmp;

			tmp = _mm256_max_epu8(result5, result6);
			result5 = _mm256_min_epu8(result5, result6);
			result6 = tmp;

			tmp = _mm256_max_epu8(result9, result10);
			result9 = _mm256_min_epu8(result9, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result11, result12);
			result11 = _mm256_min_epu8(result11, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result13, result14);
			result13 = _mm256_min_epu8(result13, result14);
			result14 = tmp;

			tmp = _mm256_max_epu8(result17, result18);
			result17 = _mm256_min_epu8(result17, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result19, result20);
			result19 = _mm256_min_epu8(result19, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result21, result22);
			result21 = _mm256_min_epu8(result21, result22);
			result22 = tmp;


			//Section 7
			tmp = _mm256_max_epu8(result0, result8);
			result0 = _mm256_min_epu8(result0, result8);
			result8 = tmp;

			tmp = _mm256_max_epu8(result1, result9);
			result1 = _mm256_min_epu8(result1, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result2, result10);
			result2 = _mm256_min_epu8(result2, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result3, result11);
			result3 = _mm256_min_epu8(result3, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result4, result12);
			result4 = _mm256_min_epu8(result4, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result5, result13);
			result5 = _mm256_min_epu8(result5, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result6, result14);
			result6 = _mm256_min_epu8(result6, result14);
			result14 = tmp;

			result7 = _mm256_min_epu8(result7, result15);
			
			tmp = _mm256_max_epu8(result16, loaded_value24);
			result16 = _mm256_min_epu8(result16, loaded_value24);
			__m256i result24 = tmp;

			//Section 8
			tmp = _mm256_max_epu8(result4, result8);
			result4 = _mm256_min_epu8(result4, result8);
			result8 = tmp;

			tmp = _mm256_max_epu8(result5, result9);
			result5 = _mm256_min_epu8(result5, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result6, result10);
			result6 = _mm256_min_epu8(result6, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result7, result11);
			result7 = _mm256_min_epu8(result7, result11);
			result11 = tmp;

			tmp = _mm256_max_epu8(result20, result24);
			result20 = _mm256_min_epu8(result20, result24);
			result24 = tmp;

			//Section 9
			tmp = _mm256_max_epu8(result2, result4);
			result2 = _mm256_min_epu8(result2, result4);
			result4 = tmp;

			tmp = _mm256_max_epu8(result3, result5);
			result3 = _mm256_min_epu8(result3, result5);
			result5 = tmp;

			tmp = _mm256_max_epu8(result6, result8);
			result6 = _mm256_min_epu8(result6, result8);
			result8 = tmp;

			tmp = _mm256_max_epu8(result7, result9);
			result7 = _mm256_min_epu8(result7, result9);
			result9 = tmp;

			tmp = _mm256_max_epu8(result10, result12);
			result10 = _mm256_min_epu8(result10, result12);
			result12 = tmp;

			tmp = _mm256_max_epu8(result11, result13);
			result11 = _mm256_min_epu8(result11, result13);
			result13 = tmp;

			tmp = _mm256_max_epu8(result18, result20);
			result18 = _mm256_min_epu8(result18, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result19, result21);
			result19 = _mm256_min_epu8(result19, result21);
			result21 = tmp;

			tmp = _mm256_max_epu8(result22, result24);
			result22 = _mm256_min_epu8(result22, result24);
			result24 = tmp;

			//Section 10
			tmp = _mm256_max_epu8(result1, result2);
			result1 = _mm256_min_epu8(result1, result2);
			result2 = tmp;

			tmp = _mm256_max_epu8(result3, result4);
			result3 = _mm256_min_epu8(result3, result4);
			result4 = tmp;

			tmp = _mm256_max_epu8(result5, result6);
			result5 = _mm256_min_epu8(result5, result6);
			result6 = tmp;

			tmp = _mm256_max_epu8(result7, result8);
			result7 = _mm256_min_epu8(result7, result8);
			result8 = tmp;

			tmp = _mm256_max_epu8(result9, result10);
			result9 = _mm256_min_epu8(result9, result10);
			result10 = tmp;

			tmp = _mm256_max_epu8(result11, result12);
			result11 = _mm256_min_epu8(result11, result12);
			result12 = tmp;

			result13 = _mm256_min_epu8(result13, result14);
			
			tmp = _mm256_max_epu8(result17, result18);
			result17 = _mm256_min_epu8(result17, result18);
			result18 = tmp;

			tmp = _mm256_max_epu8(result19, result20);
			result19 = _mm256_min_epu8(result19, result20);
			result20 = tmp;

			tmp = _mm256_max_epu8(result21, result22);
			result21 = _mm256_min_epu8(result21, result22);
			result22 = tmp;

			tmp = _mm256_max_epu8(result23, result24);
			result23 = _mm256_min_epu8(result23, result24);
			result24 = tmp;

			//Section 11
						
			result16 = _mm256_max_epu8(result0, result16);
						
			result17 = _mm256_max_epu8(result1, result17);
						
			result18 = _mm256_max_epu8(result2, result18);
						
			result19 = _mm256_max_epu8(result3, result19);
						
			result20 = _mm256_max_epu8(result4, result20);
						
			result21 = _mm256_max_epu8(result5, result21);
						
			result6 = _mm256_min_epu8(result6, result22);
						
			result7 = _mm256_min_epu8(result7, result23);
						
			result8 = _mm256_min_epu8(result8, result24);
			
			//Section 12
						
			result16 = _mm256_max_epu8(result8, result16);
						
			result17 = _mm256_max_epu8(result9, result17);
						
			result10 = _mm256_min_epu8(result10, result18);
				
			result11 = _mm256_min_epu8(result11, result19);
				
			result12 = _mm256_min_epu8(result12, result20);
					
			result13 = _mm256_min_epu8(result13, result21);
		
			//Section 13
			
			result10 = _mm256_max_epu8(result6, result10);
						
			result11 = _mm256_max_epu8(result7, result11);
						
			result12 = _mm256_min_epu8(result12, result16);
						
			result13 = _mm256_min_epu8(result13, result17);
			
			//Section 14
			
			result12 = _mm256_max_epu8(result10, result12);
			
			result11 = _mm256_min_epu8(result11, result13);
			
			//Last Section
			result12 = _mm256_max_epu8(result11, result12);

			int pixel_dst = H * imgWidth * 3 + W;
			_mm256_storeu_si256((__m256i*)(imgDst + pixel_dst), result12);
		};
	};
}