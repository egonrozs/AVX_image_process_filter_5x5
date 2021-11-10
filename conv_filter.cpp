#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"

void conv_filter(int imgHeight, int imgWidth, int imgWidthF,
				 short *filter, unsigned char *imgSrcExt, unsigned char *imgDst)
{
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < imgWidth; W++)
		{
			for (int fy = 0; fy < 5; fy++)
			{
				for (int fx = 0; fx < 5; fx++)
				{
					for (int rgb = 0; rgb < 3; rgb++)
					{
						int acc[3];
						if (fx == 0 && fy == 0)
							acc[rgb] = 0;
						acc[rgb] = acc[rgb] + imgSrcExt[3*((H + fy) * imgWidthF + (fx + W))+rgb] * filter[(fy * 5 + fx)];
						if (fx == 4 && fy == 4)
						{
							if (acc[rgb] > 255)
								acc[rgb] = 255;
							else if (acc[rgb] < 0)
								acc[rgb] = 0;
							imgDst[3 * (H * imgWidth + W) + rgb] = (char)acc[rgb];
						};
					};
				};
			};
		};
	};
}
