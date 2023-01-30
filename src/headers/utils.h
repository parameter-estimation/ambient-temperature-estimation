#ifndef UTILS_H
#define UTILS_H

#include <iostream>

void instrument_array(double * values, int length);
std::vector<double> build_vector(double *buffer, int length);
void filter_data(double *data, int length, int window_length);

#endif
