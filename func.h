void conv_filter(int imgHeight, int imgWidth, int imgWidthF,
	short *filter, unsigned char *imgSrcExt, unsigned char *imgDst);

void conv_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	short *filter, unsigned char *imgSrcExt, unsigned char *imgDst);

void conv_filter_avx_preconv(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
	short *filter, unsigned char *imgSrcExt, unsigned char *imgDst, short *imgDstConv);

void conv_filter_avx_sh(int imgHeight, int imgWidth, int imgWidthF,
	short *filter, unsigned char *imgSrcExt, unsigned char *imgDst);

void median_filter(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst);

void median_filter_avx(int imgHeight, int imgWidth, int imgWidthF,
	unsigned char* imgSrcExt, unsigned char* imgDst);

void copy(int imgHeight, int imgWidth, int imgWidthF,
	short* filter, unsigned char* imgSrcExt, unsigned char* imgDst);

