
#ifndef OPTIMIZER_API_H
#define OPTIMIZER_API_H

#include "optimizer_types.h"

class optimizer_api {

public:
    void init(optimizer_settings_t settings);
    void feed(double t, std::vector<double> *x);
    optimizer_result_t fit();
    modeled_state_timeseries_t solve(optimizer_model_params_t *params);
};

#endif
