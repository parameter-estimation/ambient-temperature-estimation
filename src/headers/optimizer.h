#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "optimizer_types.h"
#include <cmath>
#include <utility>

using namespace std;
typedef vector< double > state_type;


class Optimizer {

    public:
    optimizer_state_buffer_t m_data_buffer;
    optimizer_settings_t m_settings;

    void solve(optimizer_model_params_t *params, modeled_state_timeseries_t *generated_data);
    void feed(double t, vector<double> *x);
    optimizer_result_t fit();

    virtual string get_model_string() = 0;
    virtual void optimizer_model_diffeq(optimizer_model_params_t *m_params, state_type &x , state_type &dxdt , double t ) = 0;
    virtual double compute_row_sse(int index, std::vector<double> modeled_state) = 0;
    virtual int get_num_modeled_dimensions() = 0;
    virtual int get_required_state_dimensions() = 0;
    virtual int get_required_initial_guess_dimensions() = 0;
    virtual int get_required_fixed_param_dimensions() = 0;
    virtual void map_param_struct_to_array(optimizer_model_params_t *params, double *values) = 0;
    virtual optimizer_model_params_t map_param_array_to_struct(const double *params) = 0;


    void init(optimizer_settings_t settings) {
        m_data_buffer.rows.clear();
        m_settings = std::move(settings);
    }

    void validate_dimensionality() {
        if (!m_data_buffer.rows.empty() && (int)m_data_buffer.rows[0].x.size() != get_required_state_dimensions()) {
            printf("Warning, unexpected state dimensionality, found %zu, expected %i\n", m_data_buffer.rows[0].x.size(), get_required_state_dimensions());
        }
        if ((int)m_settings.initial_guesses.size() != get_required_initial_guess_dimensions()) {
            printf("Warning, unexpected initial guess dimensionality, found %zu, expected %i\n", m_settings.initial_guesses.size(), get_required_initial_guess_dimensions());
        }
        if ((int)m_settings.fixed_parameters.size() != get_required_fixed_param_dimensions()) {
            printf("Warning, unexpected fixed parameter dimensionality, found %zu, expected %i\n", m_settings.fixed_parameters.size(), get_required_fixed_param_dimensions());
        }
    }

    protected:

    virtual void get_parameter_guesses(double *guesses) = 0;
    virtual void fill_initial_state(state_type *state, optimizer_model_params_t *params) = 0;
    virtual int get_num_free_parameters() = 0;
    virtual void get_param_lower_bounds(double *bounds) = 0;
    virtual void get_param_upper_bounds(double *bounds) = 0;
    virtual void get_step_sizes(double *bounds) = 0;

    void numeric_solver(optimizer_model_params_t *params, modeled_state_timeseries_t *generated_data);

};

class OptimizerModel {
private:
    optimizer_model_params_t *m_params;
    Optimizer *m_instance;
public:
    explicit OptimizerModel(optimizer_model_params_t *params, Optimizer *instance ) {
        m_params = params;
        m_instance = instance;
    }
    void operator() ( state_type &x , state_type &dxdt , double t ) {
        m_instance->optimizer_model_diffeq(m_params, x, dxdt, t);
    }
};


#endif
