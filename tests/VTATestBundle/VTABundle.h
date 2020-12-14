#ifndef VTALIB_EXAMPLE_VTABUNDLE_H
#define VTALIB_EXAMPLE_VTABUNDLE_H


void xlnk_reset();

int convolution(int8_t* input, int8_t *kernel, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w);






#endif //VTALIB_EXAMPLE_VTABUNDLE_H
