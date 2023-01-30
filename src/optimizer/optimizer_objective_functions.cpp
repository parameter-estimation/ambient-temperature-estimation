
#include "optimizer.h"
#include <cmath>       /* sqrt */
#include "objective_functions.h"



double compute_sse(modeled_state_timeseries_t *current_data, objective_function_extra_t *extra)
{
    double lossStatistic = 0;
    int length = (int)current_data->t.size();
    for (int ii = 0; ii < length; ii++) {
        lossStatistic += extra->instance->compute_row_sse(ii, current_data->x[ii]);
    }
    return lossStatistic;
}

double optimizer_objective_function(unsigned n, const double *current_params, double *grad, void *extra_data)
{
    auto *extra = (objective_function_extra_t *) extra_data;
    modeled_state_timeseries_t current_data;
    optimizer_model_params_t params = extra->instance->map_param_array_to_struct(current_params);
    current_data.t = std::vector<double>(extra->instance->m_data_buffer.rows.size());
    current_data.x = vector<vector<double>>(extra->instance->m_data_buffer.rows.size(), vector<double>(extra->instance->get_num_modeled_dimensions()));
    extra->instance->solve(&params, &current_data);
    double sse = compute_sse(&current_data, extra);
    extra->step_count++;
    if (extra->instance->m_settings.verbose && (extra->step_count < 100 || extra->step_count % 100 == 0)) {
        double rmse = sqrt(sse/(int)extra->instance->m_data_buffer.rows.size());
        printf("%i: rmse:%.5f h:%.6f q:%.6f T_amb_0:%.6f T_dev_0:%.6f\n",
               extra->step_count, rmse, params.h, params.q, params.T_amb_0, params.T_dev_0);
    }
    return lossStatistic;
}
