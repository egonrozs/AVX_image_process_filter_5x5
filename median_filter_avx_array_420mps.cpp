#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include <iostream>
#include "stdio.h"

#include "defs.h"

#define UNROLL 1

void median_filter_avx(int* change_index,int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst)
{

	__m256i result[25];
	__m256i tmp;
#pragma omp parallel private(result,tmp)
#pragma omp for schedule(dynamic,1)nowait
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < 3 * imgWidth; W += 32)
		{

			for (int fy = 0; fy < 5; fy++)
			{
				for (int fx = 0; fx < 5; fx++)
				{

					result[5 * fy + fx] = _mm256_loadu_si256((__m256i const*) & imgSrcExt[3 * (H + fy) * imgWidthF + (3 * fx + W)]);

				};
			};

			for (int i = 0; i < 226; i += 2)
			{
				tmp = _mm256_max_epu8(result[change_index[i]], result[change_index[i + 1]]);
				result[change_index[i]] = _mm256_min_epu8(result[change_index[i]], result[change_index[i + 1]]);
				result[change_index[i + 1]] = tmp;
			};

			int pixel_dst = H * imgWidth * 3 + W;
			_mm256_storeu_epi8(imgDst + pixel_dst, result[12]);
		};
	};
}