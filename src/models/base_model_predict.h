#ifndef BASE_MODEL_PREDICT_H
#define BASE_MODEL_PREDICT_H

#include "optimizer.h"

#define TEMPERATURE_SEARCH_WINDOW_HALF 1

class BaseModelPredict: public Optimizer {

    using Optimizer::Optimizer;

    std::string get_model_string() override {
        return "base_predict";
    }

    int get_num_free_parameters() override { return 2; }
    int get_num_modeled_dimensions() override { return 1; }
    int get_required_state_dimensions() override { return 1; }
    int get_required_initial_guess_dimensions() override { return 1; }
    int get_required_fixed_param_dimensions() override { return 2; }

    optimizer_model_params_t map_param_array_to_struct(const double *params) override {
        return {
//            .h = params[0],
//            .q = params[1],
            .T_dev_0 = params[0],
            .T_amb_0 = params[1]
        };
    }

    void map_param_struct_to_array(optimizer_model_params_t *params, double *values) override {
        values[0] = params->T_dev_0;
        values[1] = params->T_amb_0;
    }

    void fill_initial_state(state_type *state, optimizer_model_params_t *params) override {
        (*state)[0] = params->T_dev_0;
    }

    void get_parameter_guesses(double *guesses) override {
        optimizer_model_params_t params = {
            .T_dev_0 = m_data_buffer.rows[0].x[0],
            .T_amb_0 = m_settings.initial_guesses[0]
        };
        map_param_struct_to_array(&params, guesses);
    }

    void get_param_lower_bounds(double *bounds) override {
        optimizer_model_params_t params = {
                .T_dev_0 = m_data_buffer.rows[0].x[0] - TEMPERATURE_SEARCH_WINDOW_HALF,
                .T_amb_0 = -30
        };
        map_param_struct_to_array(&params, bounds);
    }
    void get_param_upper_bounds(double *bounds) override {
        optimizer_model_params_t params = {
                .T_dev_0 = m_data_buffer.rows[0].x[0] + TEMPERATURE_SEARCH_WINDOW_HALF,
                .T_amb_0 = 70
        };
        map_param_struct_to_array(&params, bounds);
    }
    void get_step_sizes(double *steps) override {
        optimizer_model_params_t params = {
                .T_dev_0 = .1,
                .T_amb_0 = 1
        };
        map_param_struct_to_array(&params, steps);
    }

    // dxdt[0] := d(T_dev)/dt
    // x[0] := T_dev(t)
    //
    // double dTdev = h * (Tamb - Tdev) + q;

    void optimizer_model_diffeq(optimizer_model_params_t *m_params, state_type &x , state_type &dxdt , double t ) override {
        double Tamb = m_params->T_amb_0;
        dxdt[0] = m_settings.fixed_parameters[0] * (Tamb - x[0]) + m_settings.fixed_parameters[1];
    }

    double compute_row_sse(int index, std::vector<double> modeled_state) override {
        double residual = modeled_state[0] - m_data_buffer.rows[index].x[0];
        return residual * residual;
    }


};

#endif
