#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include <iostream>
#include "stdio.h"

#include "defs.h"

#define UNROLL 1
//V.BMS2 valami nem kóser a kimenettel
void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst)
{


#pragma omp parallel
#pragma omp for schedule(dynamic,1)nowait
	for (int H = 0; H < imgHeight; H += 2)
	{
		for (int W = 0; W < 3 * imgWidth; W += 32)
		{

			
			__m256i loaded_value12 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 2) * imgWidthF + (6 + W)]);
			__m256i loaded_value13 = _mm256_lddqu_si256((__m256i const*) & imgSrcExt[3 * (H + 3) * imgWidthF + (6 + W)]);
			

			int pixel_dst1 = H * imgWidth * 3 + W;
			int pixel_dst2 = (H + 1) * imgWidth * 3 + W;

			_mm256_storeu_si256((__m256i*)(imgDst + pixel_dst1), loaded_value12);
			_mm256_storeu_si256((__m256i*)(imgDst + pixel_dst2), loaded_value13);
		};
	};
}