

#include "base_model_train.h"
#include "base_model_predict.h"
#include "optimizer_api.h"

static BaseModelTrain optimizer_base_train;
static BaseModelPredict optimizer_base_predict;
static Optimizer * optimizer_singleton = &optimizer_base_train;

void optimizer_api::init(optimizer_settings_t settings) {
    if (settings.model.compare("train") == 0) {
        optimizer_singleton = &optimizer_base_train;
    } else if (settings.model.compare("predict") == 0) {
        optimizer_singleton = &optimizer_base_predict;
    } else {
        printf("Error, invalid model: %s\n", settings.model.c_str());
    }
    optimizer_singleton->init(settings);
}

void optimizer_api::feed(double t, vector<double> *x) {
    optimizer_singleton->feed(t, x);
}

optimizer_result_t optimizer_api::fit() {
    optimizer_result_t results = optimizer_singleton->fit();
    return results;
}

modeled_state_timeseries_t optimizer_api::solve(optimizer_model_params_t *params) {
    modeled_state_timeseries_t current_data;
    int length = optimizer_singleton->m_data_buffer.rows.size();
    current_data.t = std::vector<double>(length);
    current_data.x = vector<vector<double>>(length, vector<double>(optimizer_singleton->get_num_modeled_dimensions()));
    optimizer_singleton->solve(params, &current_data);
    return current_data;
}
