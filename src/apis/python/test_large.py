
import ambient_optimizer_python_api as aopa
import pandas as pd

data = pd.read_csv("../../../data/long_chamber_data.csv")
Tdev = data['Tdev']
Tamb = data['Tamb']

# Generate simulated truth data
aopa.reset()
for ii in range(len(Tdev)):
    # aopa.feed(ii, np.array([10000, T_amb_0]))
    aopa.feed(ii, [Tdev[ii], Tamb[ii]])

result = aopa.fit()
