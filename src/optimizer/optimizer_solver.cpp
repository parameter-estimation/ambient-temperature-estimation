
#include "optimizer.h"
#include <cstdio>
#include "optimizer_types.h"
#include <boost/numeric/odeint.hpp>


void Optimizer::numeric_solver(optimizer_model_params_t *params, modeled_state_timeseries_t *generated_data)
{
    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<state_type> stepper_type;

    state_type state(get_num_modeled_dimensions());
    fill_initial_state(&state, params);

    double err_abs = 1.0e-8;
    double err_rel = 1.0e-8;

    auto model = OptimizerModel(params, this);
    boost::numeric::odeint::result_of::make_dense_output< stepper_type >::type stepper = make_dense_output(err_abs, err_rel, stepper_type());

    generated_data->t[0] = 0;
    generated_data->x[0] = state;

    for (int ii=1; ii<(int)generated_data->t.size(); ii++) {
        integrate_adaptive(stepper, model, state, generated_data->t[ii-1], generated_data->t[ii], 0.1 );
        generated_data->t[ii] = ii;
        generated_data->x[ii] = state;
    }

}


void Optimizer::solve(optimizer_model_params_t *params, modeled_state_timeseries_t *generated_data)
{
    numeric_solver(params, generated_data);
}

