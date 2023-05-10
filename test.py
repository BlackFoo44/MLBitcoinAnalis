
import pandas as pd

import matplotlib.pyplot as plot





import warnings

warnings.filterwarnings("ignore")


data_frame = pd.read_csv("D:/Python stud/CryptoData/data/.coinbaseUSD.csv")

data_frame.columns = ['Timestamp', 'Weighted_Price', 'Amount']
# print(data_frame.describe())
# print(data_frame.info())
print(data_frame.head())

# # Unix-time to
# data_frame.Timestamp = pd.to_datetime(data_frame.Timestamp, unit="s")
#
# # Resampling to daily frequency
# data_frame.index = data_frame.Timestamp
# data_frame = data_frame.resample("D").mean()
# data_frame.head()
#
# # Resampling to monthly frequency
# df_month = data_frame.resample("M").mean()
#
# # Resampling to annual frequency
# df_year = data_frame.resample("A-DEC").mean()
#
# # Resampling to quarterly frequency
# df_Q = data_frame.resample("Q-DEC").mean()
#
#
# # PLOTS
# fig = plot.figure(figsize=[15, 7])
# plot.suptitle("Bitcoin exchanges, mean USD", fontsize=22)
#
# plot.subplot(221)
# plot.plot(data_frame.Weighted_Price, "-", label="By Days")
# plot.legend()
#
# plot.subplot(222)
# plot.plot(df_month.Weighted_Price, "-", label="By Months")
# plot.legend()
#
# plot.subplot(223)
# plot.plot(df_Q.Weighted_Price, "-", label="By Quarters")
# plot.legend()
#
# plot.subplot(224)
# plot.plot(df_year.Weighted_Price, "-", label="By Years")
# plot.legend()
#
# # plt.tight_layout()
# plot.show()