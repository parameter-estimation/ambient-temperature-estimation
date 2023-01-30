#ifndef OBJECTIVE_FUNCTIONS_H
#define OBJECTIVE_FUNCTIONS_H

#include "optimizer.h"

typedef struct {
    int step_count;
    Optimizer *instance;
} objective_function_extra_t;

double compute_sse(modeled_state_timeseries_t *current_data, objective_function_extra_t *extra);

double optimizer_objective_function(unsigned n, const double *current_params, double *grad, void *extra_data);

#endif
