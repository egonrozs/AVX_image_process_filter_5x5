#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"

#define UNROLL 1

void conv_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
				 short *filter, unsigned char *imgSrcExt, unsigned char *imgDst)
{
	// Az együtthatókből generálunk vektor tömböt
	__m256i filter_laplace[25];
	__m256i conv_d;
	__m256i acc;
	__m256i max= _mm256_set1_epi16(0);
	__m256i min= _mm256_set1_epi16(255);
	__m256i* acc_tmp;

	for (int k=0; k < 25; k++)
	{
		filter_laplace[k] = _mm256_set1_epi16(filter[k]);
	};

	{
		for (int H = 0; H < imgHeight; H++)
		{
			for (int W = 0; W < 3*imgWidth; W=W+16)
			{
				acc = _mm256_set1_epi16(0);
				for (int fy = 0; fy < 5; fy++)
				{
					for (int fx = 0; fx < 5; fx++)
					{
						
						conv_d = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i const*)&imgSrcExt[3*(H + fy) * imgWidthF + (3*fx + W) ]));
						acc = _mm256_add_epi16(acc,_mm256_mullo_epi16(conv_d, filter_laplace[5*fy+fx]));

					};

				};
				acc = _mm256_max_epi16(acc, max);
				acc = _mm256_min_epi16(acc, min);
				acc_tmp = &acc;
				short* output;
				output = (short*)acc_tmp;
				for (int i = 0; i < 16; i++)
				{
					short tmp = output[i];
					imgDst[3 * (H * imgWidth) + W+i] = (unsigned char)tmp;
				};
				
			};
		};
	}

	// Végiglépkedünk a kimeneti kép sorain
	//  és végiglépkedünk a kimeneti kép oszlopain, per vektor



}

