import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib as mpl
from scipy import stats
import statsmodels.api as stats_mod
from itertools import product
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


data_frame = pd.read_csv("D:/Python stud/CryptoData/data/BTC-USD.csv")
# data_frame = pd.read_csv("D:/Python stud/CryptoData/data/bitcoin_kaggle.csv")
# data_frame.columns = ['Timestamp', 'Weighted_Price', 'Amount']
print(data_frame.head())


print(data_frame.describe())

print(data_frame.info())

# Unix-time to
data_frame.Timestamp = pd.to_datetime(data_frame.Timestamp)

# Resampling to daily frequency
data_frame.index = data_frame.Timestamp
data_frame = data_frame.resample("D").mean()
data_frame.head()

# Resampling to monthly frequency
data_frame_month = data_frame.resample("M").mean()

# Resampling to annual frequency
data_frame_year = data_frame.resample("A-DEC").mean()

# Resampling to quarterly frequency
data_frame_Q = data_frame.resample("Q-DEC").mean()


# PLOTS
fig = plot.figure(figsize=[15, 7])
plot.suptitle("Bitcoin exchanges, mean USD", fontsize=22)

plot.subplot(221)
plot.plot(data_frame.Weighted_Price, "-", label="By Days")
plot.legend()

plot.subplot(222)
plot.plot(data_frame_month.Weighted_Price, "-", label="By Months")
plot.legend()

plot.subplot(223)
plot.plot(data_frame_Q.Weighted_Price, "-", label="By Quarters")
plot.legend()

plot.subplot(224)
plot.plot(data_frame_year.Weighted_Price, "-", label="By Years")
plot.legend()

# plt.tight_layout()
plot.show()

# Stationarity check and STL-decomposition of the serie
plot.figure(figsize=[15, 7])
stats_mod.tsa.seasonal_decompose(data_frame_month.Weighted_Price).plot()
print(
    "Dickey–Fuller test: p=%f" % stats_mod.tsa.stattools.adfuller(data_frame_month.Weighted_Price)[1]
)
plot.show()
#
# Box-Cox Transformations
data_frame_month["Weighted_Price_box"], lmbda = stats.boxcox(data_frame_month.Weighted_Price)
print(
    "Dickey–Fuller test: p=%f" % stats_mod.tsa.stattools.adfuller(data_frame_month.Weighted_Price)[1]
)


# Seasonal differentiation
data_frame_month[
    "prices_box_diff"
] = data_frame_month.Weighted_Price_box - data_frame_month.Weighted_Price_box.shift(12)
print(
    "Dickey–Fuller test: p=%f"
    % stats_mod.tsa.stattools.adfuller(data_frame_month.prices_box_diff[12:])[1]
)


# Regular differentiation
data_frame_month[
    "prices_box_diff2"
] = data_frame_month.prices_box_diff - data_frame_month.prices_box_diff.shift(1)
plot.figure(figsize=(15, 7))

# STL-decomposition
stats_mod.tsa.seasonal_decompose(data_frame_month.prices_box_diff2[13:]).plot()
print(
    "Dickey–Fuller test: p=%f"
    % stats_mod.tsa.stattools.adfuller(data_frame_month.prices_box_diff2[13:])[1]
)

plot.show()


# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plot.figure(figsize=(15, 7))
ax = plot.subplot(211)
stats_mod.graphics.tsa.plot_acf(
    data_frame_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax
)

ax = plot.subplot(212)
stats_mod.graphics.tsa.plot_pacf(
    data_frame_month.prices_box_diff2[13:].values.squeeze(), lags=26, ax=ax
)
plot.tight_layout()
plot.show()


# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings("ignore")
for param in parameters_list:
    try:
        model = stats_mod.tsa.statespace.SARIMAX(
            data_frame_month.Weighted_Price_box,
            order=(param[0], d, param[1]),
            seasonal_order=(param[2], D, param[3], 12),
        ).fit(disp=-1)
    except ValueError:
        print("wrong parameters:", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])


# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ["parameters", "aic"]
print(result_table.sort_values(by="aic", ascending=True).head())
print(best_model.summary())


# STL-decomposition
plot.figure(figsize=(15, 7))
plot.subplot(211)
best_model.resid[13:].plot()
plot.ylabel("Residuals")
ax = plot.subplot(212)
stats_mod.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Dickey–Fuller test:: p=%f" % stats_mod.tsa.stattools.adfuller(best_model.resid[13:])[1])

plot.tight_layout()
plot.show()


# Inverse Box-Cox Transformation Function
def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)


df_month2 = data_frame_month[["Weighted_Price"]]
date_list = [
    datetime(2023, 5, 31),
    datetime(2023, 6, 30),
    datetime(2023, 7, 31),
    datetime(2023, 8, 31),
    datetime(2023, 9, 30),
    datetime(2023, 10, 31),
    datetime(2023, 11, 30),
    datetime(2023, 12, 31),
    datetime(2024, 1, 31),
    datetime(2024, 2, 28),
    datetime(2024, 3, 31),
    datetime(2024, 4, 30),
    datetime(2024, 5, 31),
    datetime(2024, 6, 30),
    datetime(2024, 7, 31),
    datetime(2024, 8, 31),
    datetime(2024, 9, 30),
    datetime(2024, 10, 31),
    datetime(2024, 11, 30),
    datetime(2024, 12, 31),
]
future = pd.DataFrame(index=date_list, columns=data_frame_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2["forecast"] = invboxcox(best_model.predict(start=0, end=180), lmbda)
plot.figure(figsize=(20, 7))
df_month2.Weighted_Price.plot()
df_month2.forecast.plot(color="r", ls="--", label="Predicted Weighted_Price")
plot.legend()
plot.title("Bitcoin exchanges, by months")
plot.ylabel("mean USD")
plot.show()
