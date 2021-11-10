#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"

void copy(int imgHeight, int imgWidth, int imgWidthF,
				 short *filter, unsigned char *imgSrcExt, unsigned char *imgDst)
{
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < imgWidth; W++)
		{
			
			for (int rgb = 0; rgb < 3; rgb++)
			{
						
				imgDst[3 * (H * imgWidth + W) + rgb] = imgSrcExt[3 * ((H + 2) * imgWidthF + (2 + W)) + rgb];
					
			};
		};	
	};
}
