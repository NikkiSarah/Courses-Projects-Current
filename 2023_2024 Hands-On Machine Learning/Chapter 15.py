#%% forecasting a time series
import tensorflow.keras as tfk
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

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
df_monthly = df.drop('day_type', axis=1).resample("M").mean()
rolling_avg_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots()
df_monthly[period].plot(ax=ax, marker=".")
rolling_avg_12_months.plot(ax=ax, legend=False)

df_monthly.diff(12)[period].plot(marker=".")

# the ARMA model family
origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_series, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
model = model.fit()
y_pred = model.forecast()
print(y_pred)

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []
for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
    model = model.fit()
    y_pred = model.forecast()[0]
    y_preds.append(y_pred)
    
y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean()
print(mae)

# preparing the data for ML models
my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tfk.utils.timeseries_dataset_from_array(
    my_series,
    targets=my_series[3:],
    sequence_length=3,
    batch_size=2
    )
print(list(my_dataset))

for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
    for ele in window_dataset:
        print(f"{ele}", end=" ")
    print()

dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))
for window_tensor in dataset:
    print(f"{window_tensor}")
    
def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_ds: window_ds.batch(length))
    return dataset

dataset = to_windows(tf.data.Dataset.range(6), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
print(list(dataset.batch(2)))

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_val = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6

seq_length = 56
train_ds = tfk.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42)
val_ds = tfk.utils.timeseries_dataset_from_array(
    rail_val.to_numpy(),
    targets=rail_val[seq_length:],
    sequence_length=seq_length,
    batch_size=32)

# forecasting using a linear model
tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.Dense(1, input_shape=[seq_length])
    ])
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                restore_best_weights=True)
optimiser = tfk.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
history = model.fit(train_ds, validation_data=val_ds, epochs=500,
                    callbacks=[early_stopping_cb])

val_loss, val_mae = model.evaluate(val_ds)
print(val_mae * 1e6)

# forecasting using a simple RNN
tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.SimpleRNN(1, input_shape=[None, 1])
    ])
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                restore_best_weights=True)
optimiser = tfk.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
history = model.fit(train_ds, validation_data=val_ds, epochs=500,
                    callbacks=[early_stopping_cb])

val_loss, val_mae = model.evaluate(val_ds)
print(val_mae * 1e6)


tfk.backend.clear_session()
tf.random.set_seed(42)
univar_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 1]),
    tfk.layers.Dense(1)
    ])
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                restore_best_weights=True)
optimiser = tfk.optimizers.SGD(learning_rate=0.02, momentum=0.9)
univar_model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
history = univar_model.fit(train_ds, validation_data=val_ds, epochs=500,
                           callbacks=[early_stopping_cb])

val_loss, val_mae = univar_model.evaluate(val_ds)
print(val_mae * 1e6)

# forecasting using a deep RNN
tfk.backend.clear_session()
tf.random.set_seed(42)
deep_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tfk.layers.SimpleRNN(32, return_sequences=True),
    tfk.layers.SimpleRNN(32),
    tfk.layers.Dense(1)
    ])
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                restore_best_weights=True)
optimiser = tfk.optimizers.SGD(learning_rate=0.02, momentum=0.9)
deep_model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
history = deep_model.fit(train_ds, validation_data=val_ds, epochs=500,
                         callbacks=[early_stopping_cb])

val_loss, val_mae = deep_model.evaluate(val_ds)
print(val_mae * 1e6)

# forecasting multivariate time series

# forecasting several time steps ahead

# forecasting using a sequence-to-sequence model




#%% handling long sequences

# fighting the unstable gradients problem

# tackling the short-term memory problem

#%% Coding Exercises: Exercise 9

# train a classification model for the SketchRNN dataset

#%% Coding Exercises: Exercise 10

# download the Bach chorales datset and unzip it

# train a model - recurrent, convlutional or both - that can predict the next time step
# (4 notes), given a sequence of time steps from a chorale

# use the model to generate Bach-like music, one note at a time:
# - give the model the start of a chorale and predict the next note

# - append it to the input sequence

# - predict the next note and append that to the sequence

# - repeat twice more
