
#include "optimizer_api.h"
#include <iostream>
#include <fstream>

static optimizer_api _api;

optimizer_model_params_t truth_params = {
        .h = 0.0064168987,
        .q = 0.028977304,
        .T_dev_0 = 20,
        .T_amb_0 = 30
};

optimizer_settings_t settings = {
        .model = "predict",
        .verbose = true,
        .initial_guesses = {30},
        .fixed_parameters = {truth_params.h, truth_params.q}
};

double T_amb[] = {30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30,
                  30,30,30,30,30,30,30,30,30,30};

double T_dev[] = {20.        , 20.09284807, 20.18510226, 20.27676635, 20.36784413,
                 20.45833934, 20.54825572, 20.63759695, 20.72636673, 20.81456871,
                 20.90220651, 20.98928375, 21.07580402, 21.16177087, 21.24718784,
                 21.33205846, 21.41638622, 21.50017458, 21.58342701, 21.66614692,
                 21.74833773, 21.83000282, 21.91114555, 21.99176926, 22.07187727,
                 22.15147288, 22.23055937, 22.30914   , 22.38721799, 22.46479658,
                 22.54187894, 22.61846825, 22.69456768, 22.77018034, 22.84530936,
                 22.91995783, 22.99412881, 23.06782538, 23.14105056, 23.21380736,
                 23.28609878, 23.3579278 , 23.42929738, 23.50021045, 23.57066994,
                 23.64067874, 23.71023975, 23.77935581, 23.84802979, 23.9162645 ,
                 23.98406276, 24.05142736, 24.11836106, 24.18486664, 24.25094683,
                 24.31660434, 24.38184188, 24.44666214, 24.51106779, 24.57506147,
                 24.63864583, 24.70182349, 24.76459703, 24.82696905, 24.88894212,
                 24.95051879, 25.01170159, 25.07249305, 25.13289566, 25.19291191,
                 25.25254428, 25.31179522, 25.37066717, 25.42916255, 25.48728378,
                 25.54503324, 25.60241332, 25.65942637, 25.71607475, 25.77236078,
                 25.82828679, 25.88385508, 25.93906793, 25.99392762, 26.04843641,
                 26.10259654, 26.15641025, 26.20987974, 26.26300722, 26.31579488,
                 26.36824489, 26.42035942, 26.4721406 , 26.52359057, 26.57471144,
                 26.62550533, 26.67597433, 26.7261205 , 26.77594593, 26.82545265};


void test_fit() {

    _api.init(settings);
    std::vector<double> state = std::vector<double>(1);

    for (int ii=0; ii<30; ii++) {
        state[0] = T_dev[ii];
//        state[1] = T_amb[ii];
        _api.feed(ii, &state);
    }

    optimizer_result_t result = _api.fit();
}

void test_generate() {
    int length = 100;
    std::vector<double> t(length);
    std::vector<std::vector<double>> x(length);

    _api.init(settings);

    std::vector<double> state(1);
    for (int ii=0; ii<length; ii++) {
        state[0] = T_dev[ii];
        _api.feed(ii, &state);
    }

    modeled_state_timeseries_t data = _api.solve(&truth_params);

    printf("generated data with length: %i\n", length);
    for (int ii=0; ii<length; ii++) {
        printf("%.4f ", data.x[ii][0]);
    }
    printf("\n");

}

void test_feed() {
    std::vector<double> state = std::vector<double>(1);
    state[0] = 1;
    _api.init(settings);
    _api.feed(0, &state);
}

void test_reset() {
    _api.init(settings);
}

int main() {

    printf("calling into test_reset()\n");
    test_reset();

    printf("calling into test_generate()\n");
    test_generate();

    printf("calling into test_feed()\n");
    test_feed();

    printf("calling into test_fit()\n");
    test_fit();
    printf("\n^^^ This test currently expects to see: `nlopt_optimize complete: 4` ^^^\n");


    return 0;
}