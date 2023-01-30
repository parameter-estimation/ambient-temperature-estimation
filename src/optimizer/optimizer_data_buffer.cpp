#include "optimizer.h"


void Optimizer::feed(double t, vector<double> *x)
{
    if ((int)x->size() != get_required_state_dimensions()) {
        printf("Error, invalid vector size %zu, expected %i\n", x->size(), get_required_state_dimensions());
    }
    optimizer_model_state_t row;
    row.t = t;
    vector<double> x_copy(x->size());
    for (int ii=0; ii<(int)x->size(); ii++) { x_copy[ii] = (*x)[ii];}
    row.x = x_copy;
    m_data_buffer.rows.push_back(row);
}

