
#include "optimizer.h"

#include <cmath>       /* sqrt */
#include <cstdio>
#include <vector>


void instrument_array(double * values, int length) {
    for (int ii=0; ii<length; ii++) {
        if (values[ii] >= 0) {
            printf(" ");
        }
        printf("%0.4f ", values[ii]);
    }
}

std::vector<double> build_vector(double *buffer, int length) {
    std::vector<double> vec = std::vector<double>(length);
    for (int ii=0; ii<length; ii++) {
        vec[ii] = buffer[ii];
    }
    return vec;
}
