#ifndef OPTIMIZER_TYPES_H
#define OPTIMIZER_TYPES_H

#include "utils.h"
#include <vector>

typedef struct {
    double h;
    double q;
    double T_dev_0;
    double T_amb_0;
} optimizer_model_params_t;

typedef struct {
    double maxtime_sec;
} nlopt_settings_t;

typedef struct {
    std::string model;
    bool verbose;
    std::vector<double> initial_guesses;
    std::vector<double> fixed_parameters;
    nlopt_settings_t nlopt;
} optimizer_settings_t;

typedef struct {
    double t;
    std::vector<double> x;
} optimizer_model_state_t;

typedef struct {
    std::vector<optimizer_model_state_t> rows;
} optimizer_state_buffer_t;

typedef struct {
    std::vector<double> t;
    std::vector<std::vector<double>> x;
} modeled_state_timeseries_t;

typedef struct {
    double rmse;
    int icount;
    optimizer_model_params_t fitted_params;
    int ifault;
    int ready_to_stop;
    bool is_valid;
    std::string model;
} optimizer_result_t;

#endif
