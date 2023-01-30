
import ambient_optimizer_python_api as aopa
import numpy as np

# Set truth parameters
h = 0.1
q = 0.001
T_dev_0 = 20
T_amb_0 = 30

# Generate simulated truth data
aopa.init({
    "model": "train",
    "verbose": True,
    # "initial_guesses": [2]
})
for ii in range(100):
    # aopa.feed(ii, np.array([10000, T_amb_0]))
    aopa.feed(ii, [10000, T_amb_0])

result = aopa.generate(h, q, T_dev_0)



# result = aopa.generate(100, h, q, T_dev_0, T_amb_0)
orig_data_T_dev = result['x'][:,0]
# orig_data_T_amb = result['x'][:,1]
orig_data_t = result['t']

# print("Generated: {}".format(result))


# Create simulated measurements by adding noise
noise_factor = 0
noise_x = np.random.normal(0, noise_factor, len(orig_data_T_dev))
measurements_T_dev = orig_data_T_dev + noise_x
noise_y = np.random.normal(0, noise_factor, len(orig_data_T_dev))
measurements_T_amb = T_amb_0 + noise_y


# feed and fit measurements
aopa.init({
    "model": "train",
    "verbose": True,
    # "initial_guesses": [2]
})
for t, T_dev in enumerate(measurements_T_dev):
    T_amb = measurements_T_amb[t]
    # x = np.array([T_dev, T_amb])
    x = [T_dev, T_amb]
    print(t, x)
    aopa.feed(t, x)

fit_result = aopa.fit()

print(fit_result)