#%% forecasting a time series
import tensorflow.keras as tfk
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

tfk.utils.get_file(
    "ridership.tgz",
    "https://github.com/ageron/data/raw/main/ridership.tgz",
    cache_dir=".",
    extract=True
)

path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()
df.head()

df["2019-03":"2019-05"].plot(marker=".", )
plt.legend(loc="lower left")

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True)
df.plot(ax=axs[0], legend=False, marker=".")
df.shift(7).plot(ax=axs[0], legend=False, linestyle=":")
diff_7.plot(ax=axs[1], marker=".")

print(list(df.loc["2019-05-25":"2019-05-27"]["day_type"]))
print(diff_7.abs().mean())

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
print((diff_7 / targets).abs().mean())

period = slice("2001", "2019")
df_monthly = df.resample("M").mean()
rolling_avg_12_months = df_monthly[period].rolling(window=12).mean()

fix, ax = plt.subplots()
df_monthly[period].plot(ax=ax, marker=".")
rolling_avg_12_months.plot(ax=ax, legend=False)

#%% handling long sequences

#%% Coding Exercises: Exercise 9

#%% Coding Exercises: Exercise 10