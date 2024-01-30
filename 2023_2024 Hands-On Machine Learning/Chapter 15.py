#%% forecasting a time series
import tensorflow.keras as tfk
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import numpy as np

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

def fit_and_evaluate_model(model, train_set, val_set, learning_rate, epochs=500):
    early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                    restore_best_weights=True)
    optimiser = tfk.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
    history = model.fit(train_set, validation_data=val_set, epochs=epochs,
                        callbacks=[early_stopping_cb])
    val_loss, val_mae = model.evaluate(val_set)
    print(val_mae * 1e6)
    
fit_and_evaluate_model(model, train_ds, val_ds, learning_rate=0.02)


tfk.backend.clear_session()
tf.random.set_seed(42)
univar_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 1]),
    tfk.layers.Dense(1)
    ])

fit_and_evaluate_model(univar_model, train_ds, val_ds, learning_rate=0.05)

# forecasting using a deep RNN
tfk.backend.clear_session()
tf.random.set_seed(42)
deep_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tfk.layers.SimpleRNN(32, return_sequences=True),
    tfk.layers.SimpleRNN(32),
    tfk.layers.Dense(1)
    ])

fit_and_evaluate_model(deep_model, train_ds, val_ds, learning_rate=0.01)

# forecasting multivariate time series
df_mulvar = df[["bus", "rail"]] / 1e6
df_mulvar["next_day_type"] = df.day_type.shift(-1)
df_mulvar = pd.get_dummies(df_mulvar, dtype=float)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_val = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]

train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
val_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=mulvar_val["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

tfk.backend.clear_session()
mulvar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(1)
])

fit_and_evaluate_model(mulvar_model, train_mulvar_ds, val_mulvar_ds, learning_rate=0.05)
rail_naive = mulvar_val["rail"].shift(7)[seq_length:]
rail_target = mulvar_val["rail"][seq_length:]
print((rail_target - rail_naive).abs().mean() * 1e6)


train_multask_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42)
val_multask_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=mulvar_val[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32)

tfk.backend.clear_session()
tf.random.set_seed(42)
multask_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 5]),
    tfk.layers.Dense(2)])

fit_and_evaluate_model(multask_model, train_multask_ds, val_multask_ds,
                       learning_rate=0.02)
y_preds_val = multask_model.predict(val_multask_ds)
for idx, name in enumerate(["bus", "rail"]):
    mae = 1e6 * tfk.metrics.mean_absolute_error(mulvar_val[name][seq_length:],
                                                y_preds_val[:, idx])
    print(name, int(mae))


# forecasting several time steps ahead
X = rail_val.to_numpy()[np.newaxis, :seq_length, np.newaxis]
for step_ahead in range(14):
    y_pred_one = univar_model.predict(X)
    X = np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)

def split_inputs_and_targets(mulvar_series, ahead=14, target_col=1):
    return mulvar_series[:, :-ahead], mulvar_series[:, -ahead:, target_col]

ahead_train_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32).map(split_inputs_and_targets)
ahead_val_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32).map(split_inputs_and_targets)

tfk.backend.clear_session()
tf.random.set_seed(42)
ahead_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(ahead_model, ahead_train_ds, ahead_val_ds, learning_rate=0.02)

X = mulvar_val.to_numpy()[np.newaxis, :seq_length]
y_pred = ahead_model.predict(X)
print(y_pred)

# forecasting using a sequence-to-sequence model
my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)
print(list(dataset))

dataset = dataset.map(lambda S: (S[:, 0], S[:, 1:]))
print(list(dataset))

def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1, batch_size=32,
                       shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size)

seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_val = to_seq2seq_dataset(mulvar_val)

tfk.backend.clear_session()
tf.random.set_seed(42)
seq2seq_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])
fit_and_evaluate_model(seq2seq_model, seq2seq_train, seq2seq_val, learning_rate=0.1)

X = mulvar_val.to_numpy()[np.newaxis, :seq_length]
y_pred_14 = seq2seq_model.predict(X)[0, -1]

y_pred_val = seq2seq_model.predict(seq2seq_val)
for ahead in range(14):
    preds = pd.Series(y_pred_val[:-1, -1, ahead],
                      index=mulvar_val.index[56 + ahead : -14 + ahead])
    mae = (preds - mulvar_val["rail"]).abs().mean() * 1e6
    print(f"MAE for +{ahead + 1}: {mae:,.0f}")

#%% handling long sequences

# fighting the unstable gradients problem
class LNSimpleRNNCell(tfk.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = tfk.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = tfk.layers.LayerNormalization()
        self.activation = tfk.activations.get(activation)
        
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

tfk.backend.clear_session()
tf.random.set_seed(42)
custom_ln_model= tfk.Sequential([
    tfk.layers.RNN(LNSimpleRNNCell(32), return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(custom_ln_model, seq2seq_train, seq2seq_val, learning_rate=0.1,
                       epochs=5)

# tackling the short-term memory problem
tfk.backend.clear_session()
tf.random.set_seed(42)
lstm_model = tfk.Sequential([
    tfk.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(lstm_model, seq2seq_train, seq2seq_val, learning_rate=0.1,
                       epochs=5)


tfk.backend.clear_session()
tf.random.set_seed(42)
conv_rnn_model = tfk.Sequential([
    tfk.layers.Conv1D(filters=32, kernel_size=4, strides=2, activation="relu",
                      input_shape=[None, 5]),
    tfk.layers.GRU(32, return_sequences=True),
    tfk.layers.Dense(14)
    ])

longer_train = to_seq2seq_dataset(mulvar_train, seq_length=112, shuffle=True, seed=42)
longer_val = to_seq2seq_dataset(mulvar_val, seq_length=112)
downsampled_train = longer_train.map(lambda X, Y: (X, Y[:, 3::2]))
downsampled_val = longer_val.map(lambda X, Y: (X, Y[:, 3::2]))

fit_and_evaluate_model(conv_rnn_model, downsampled_train, downsampled_val,
                       learning_rate=0.1, epochs=5)


#%% Coding Exercises: Exercise 9

# train a classification model for the SketchRNN dataset
tf_download_root = "http://download.tensorflow.org/data/"
filename = "quickdraw_tutorial_dataset_v1.tar.gz"
filepath = tf.keras.utils.get_file(filename,
                                   tf_download_root + filename,
                                   cache_dir=".",
                                   extract=True)



#%% Coding Exercises: Exercise 10

# download the Bach chorales datset and unzip it

# train a model - recurrent, convlutional or both - that can predict the next time step
# (4 notes), given a sequence of time steps from a chorale

# use the model to generate Bach-like music, one note at a time:
# - give the model the start of a chorale and predict the next note

# - append it to the input sequence

# - predict the next note and append that to the sequence

# - repeat twice more
