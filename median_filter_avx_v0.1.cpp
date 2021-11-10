#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"

void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	 unsigned char* imgSrcExt, unsigned char* imgDst)
{
	__m256i result[25];
	__m256i tmp;
	__m256i* result_pointer;
	
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < 3*imgWidth; W+=32)
		{
			for (int fy = 0; fy < 5; fy++)
			{
				for (int fx = 0; fx < 5; fx++)
				{
					
					result[5 * fy + fx] = _mm256_loadu_si256((__m256i const*)&imgSrcExt[3 * (H + fy) * imgWidthF + (3 * fx + W)]);
				
				};
			};


			for (int p = 1; p < 25; p += p)
			{
				for (int k = p; k >= 1; k = k / 2)
				{
					for (int j = k % p; j + k < 25; j += k + k)
					{
						for (int i = 0; i < k; i++)
						{
							if (((i + j) / (p + p) == (i + j + k) / (p + p)) && ((i + j + k) < 25))
							{
								tmp = _mm256_max_epu8(result[i+j], result[i+j+k]);
								result[i+j] = _mm256_min_epu8(result[i + j], result[i + j + k]);
								result[i + j + k] = tmp;
							};
						};
					};
				};	
			};
			result_pointer = &result[12];
			unsigned char* output;
			output = (unsigned char*)result_pointer;
			for (int i = 0; i < 32; i++)
			{
				imgDst[3 * (H * imgWidth) + W + i] = (unsigned char)output[i];
			};

		};
	};
}
