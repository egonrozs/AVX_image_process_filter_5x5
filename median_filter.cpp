#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"
//Snassz mezei C kód: kész
void median_filter(int imgHeight, int imgWidth, int imgWidthF,
	 unsigned char* imgSrcExt, unsigned char* imgDst)
{
	int out[3][25];
	int tmp;
	int rgb;
	for (int H = 0; H < imgHeight; H++)
	{
		for (int W = 0; W < imgWidth; W++)
		{
			for (int fy = 0; fy < 5; fy++)
			{
				for (int fx = 0; fx < 5; fx++)
				{
					for (rgb = 0; rgb < 3; rgb++)
					{
						out[rgb][5 * fy + fx] = imgSrcExt[3 * ((H + fy) * imgWidthF + (fx + W)) + rgb];
					};
				};
			};

			for (rgb = 0; rgb < 3; rgb++)
			{
				for (int p = 1; p < 25; p += p)
				{
					for (int k = p; k > 0; k = k / 2)
					{
						for (int j = k % p; j + k < 25; j += k + k)
						{
							for (int i = 0; i < k; i++)
							{
								if (((i + j) / (p + p) == (i + j + k) / (p + p))&&((i + j + k) < 25))
								{
									if (out[rgb][i + j] > out[rgb][i + j + k])
									{
										tmp = out[rgb][i + j + k];
										out[rgb][i + j + k] = out[rgb][i + j];
										out[rgb][i + j] = tmp;
									};

								};
							};	
						};
					};
				};
				imgDst[3 * (H * imgWidth + W) + rgb] = (unsigned char)out[rgb][12];
			};

		};
	};
}