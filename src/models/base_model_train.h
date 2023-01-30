#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include "optimizer.h"

#define TEMPERATURE_SEARCH_WINDOW_HALF 1

class BaseModelTrain: public Optimizer {

    using Optimizer::Optimizer;

    std::string get_model_string() override {
        return "base_train";
    }

    int get_num_free_parameters() override { return 3; }
    int get_num_modeled_dimensions() override { return 1; }
    int get_required_state_dimensions() override { return 2; }
    int get_required_initial_guess_dimensions() override { return 0; }
    int get_required_fixed_param_dimensions() override { return 0; }

    optimizer_model_params_t map_param_array_to_struct(const double *params) override {
        return {
            .h = params[0],
            .q = params[1],
            .T_dev_0 = params[2],
//            .T_amb_0 = params[3]
        };
    }

    void map_param_struct_to_array(optimizer_model_params_t *params, double *values) override {
        values[0] = params->h;
        values[1] = params->q;
        values[2] = params->T_dev_0;
//        values[3] = params->T_amb_0;
    }

    void fill_initial_state(state_type *state, optimizer_model_params_t *params) override {
        (*state)[0] = params->T_dev_0;
    }

    void get_parameter_guesses(double *guesses) override {
        optimizer_model_params_t params = {

                .h = 0.0064168987,
                .q = 0.028977304,
            .T_dev_0 = m_data_buffer.rows[0].x[0],
//            .T_amb_0 = data->x[0][1]
        };
        map_param_struct_to_array(&params, guesses);
    }

    void get_param_lower_bounds(double *bounds) override {
        optimizer_model_params_t params = {
                .h = 0,
                .q = 0,
                .T_dev_0 = m_data_buffer.rows[0].x[0] - TEMPERATURE_SEARCH_WINDOW_HALF,
//                .T_amb_0 = initial_data->x[1] - TEMPERATURE_SEARCH_WINDOW_HALF
        };
        map_param_struct_to_array(&params, bounds);
    }
    void get_param_upper_bounds(double *bounds) override {
        optimizer_model_params_t params = {
                .h = 1,
                .q = 1,
                .T_dev_0 = m_data_buffer.rows[0].x[0] + TEMPERATURE_SEARCH_WINDOW_HALF,
//                .T_amb_0 = initial_data->x[1] + TEMPERATURE_SEARCH_WINDOW_HALF
        };
        map_param_struct_to_array(&params, bounds);
    }
    void get_step_sizes(double *steps) override {
        optimizer_model_params_t params = {
                .h = 0.001,
                .q = 0.001,
                .T_dev_0 = 0.1,
//                .T_amb_0 = 0.1
        };
        map_param_struct_to_array(&params, steps);
    }

    // dxdt[0] := d(T_dev)/dt
    // x[0] := T_dev(t)
    //
    // double dTdev = h * (Tamb - Tdev) + q;

    void optimizer_model_diffeq(optimizer_model_params_t *m_params, state_type &x , state_type &dxdt , double t ) override {
        int data_index = (int)t;
        dxdt[0] = m_params->h * (m_data_buffer.rows[data_index].x[1] - x[0]) + m_params->q;
    }

    double compute_row_sse(int index, std::vector<double> modeled_state) override {
        double residual = modeled_state[0] - m_data_buffer.rows[index].x[0];
        return residual * residual;
    }


};

#endif
