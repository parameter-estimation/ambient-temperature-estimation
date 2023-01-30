#include <nlopt.h>
#include "base_model_train.h"
#include "objective_functions.h"


optimizer_result_t Optimizer::fit()
{
    // TODO: Figure out why calling this with no data causes seg fault, so we don't need this length test
    if (m_data_buffer.rows.empty()) {
        printf("Cannot fit empty buffer\n");
        optimizer_result_t result;
        result.is_valid = false;
        return result;
    }

    validate_dimensionality();

    if (m_settings.verbose) {
        printf("fitting model: %s on %i data points\n", get_model_string().c_str(), (int)m_data_buffer.rows.size());
    }

    objective_function_extra_t extra = {
            .step_count = 0,
            .instance = this
    };

    optimizer_model_state_t initial_data;
    initial_data.t = 0;
    initial_data.x = m_data_buffer.rows[0].x;

    int N = get_num_free_parameters();
    double lb[N];
    double ub[N];
    double step[N];
    double params[N];
    get_param_lower_bounds(lb);
    get_param_upper_bounds(ub);
    get_step_sizes(step);
    get_parameter_guesses(params);


    if (m_settings.verbose) {
        printf("fit lower bounds:    ");
        instrument_array(lb, N);
        printf("\n");
        printf("fit initial guesses: ");
        instrument_array(params, N);
        printf("\n");
        printf("fit upper bounds:    ");
        instrument_array(ub, N);
        printf("\n");
        printf("fit step sizes:      ");
        instrument_array(step, N);
        printf("\n");
    }

    nlopt_opt opt;
    opt = nlopt_create(NLOPT_LN_NELDERMEAD, N);
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    nlopt_set_initial_step(opt, step);
    nlopt_set_min_objective(opt, &optimizer_objective_function, &extra);
    nlopt_set_xtol_rel(opt, 1e-8);

    if (m_settings.nlopt.maxtime_sec != 0) {
        if (m_settings.verbose) {
            printf("using maxtime_sec: %f\n", m_settings.nlopt.maxtime_sec);
        }
        nlopt_set_maxtime(opt, m_settings.nlopt.maxtime_sec);
    }

    if (m_settings.verbose) {
        printf("nlopt configured, calling optimize\n");
    }

    double minimum_value;
    // cout << "starting nlopt" << std::endl;
    int nloptResult = nlopt_optimize(opt, params, &minimum_value);

    if (m_settings.verbose) {
        double rmse = sqrt(minimum_value/(int)m_data_buffer.rows.size());
        printf("nlopt_optimize complete.  result=%i icount=%i rmse=%.6f\n", nloptResult, extra.step_count, rmse);
    }

    optimizer_result_t result;
    result.icount = extra.step_count;
    result.ifault = nloptResult;
    result.is_valid = result.ifault > 0;
    result.model = get_model_string();  // including model string here was causing garbage data to come through the emscripten interface

    if (nloptResult < 0) {
        printf("nlopt failed!\n");
        printf("fit lower bounds:    ");
        instrument_array(lb, N);
        printf("\n");
        printf("fit initial guesses: ");
        instrument_array(params, N);
        printf("\n");
        printf("fit upper bounds:    ");
        instrument_array(ub, N);
        printf("\n");
        printf("fit step sizes:      ");
        instrument_array(step, N);
        printf("\n");

    } else {
        optimizer_model_params_t fitted_params = map_param_array_to_struct(params);
        result.fitted_params = fitted_params;

        // Generate data from the fitted parameters so we can compute RMSE of fit for absolute and differential data
        modeled_state_timeseries_t fitted_data;
        fitted_data.t = std::vector<double>(m_data_buffer.rows.size());
        fitted_data.x = std::vector<std::vector<double>>(m_data_buffer.rows.size(), std::vector<double>(get_num_modeled_dimensions()));
        solve(&fitted_params, &fitted_data);

        result.rmse = sqrt(compute_sse(&fitted_data, &extra)/m_data_buffer.rows.size());

    }

    nlopt_destroy(opt);

    return result;

}
