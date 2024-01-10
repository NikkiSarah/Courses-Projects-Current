# the perceptron
import os

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(int)

clf = Perceptron()
clf.fit(X, y)
y_pred = clf.predict([[2, 0.5]])

# installing tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

print(tf.__version__)

# building an image classifier using the sequential API
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_val.shape)
print(X_train_val.dtype)

X_val, X_train = X_train_val[:5000] / 255.0, X_train_val[5000:] / 255.0
y_val, y_train = y_train_val[:5000], y_train_val[5000:]
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Boot']
print(class_names[y_train[0]])

model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(300, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())
print(model.layers)
hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense_3') is hidden1)

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

pd.DataFrame(history.history).plot()
plt.gca().set_ylim(0, 1)

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=1)
print(y_pred)
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(y_new)

# building a regression MLP using the functional API
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

input_ = Input(shape=X_train.shape[1:])
hidden1 = Dense(30, activation='relu')(input_)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = Concatenate()([input_, hidden2])
output = Dense(1)(concat)
model = Model(inputs=[input_], outputs=[output])

input_A = Input(shape=[5], name="wide_input")
input_B = Input(shape=[6], name="deep_input")
hidden1 = Dense(30, activation='relu')(input_B)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = Concatenate()([input_A, hidden2])
output = Dense(1, name="output")(concat)
model = Model(inputs=[input_A, input_B], outputs=[output])
model.compile(loss='mse', optimizer=SGD(learning_rate=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_val_A, X_val_B = X_val[:, :5], X_val[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_val_A, X_val_B), y_val))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

output = Dense(1, name="main_output")(concat)
aux_output = Dense(1, name="aux_output")(hidden2)
model = Model(inputs=[input_A, input_B], outputs=[output, aux_output])
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer="sgd")

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_val_A, X_val_B], [y_val, y_val]))
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])


# using the subclassing API to build dynamic models
class WideAndDeepModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_output = Dense(1)
        self.aux_output = Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = Concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


model = WideAndDeepModel()

# saving and restoring a model
from tensorflow.keras.models import load_model

(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()
X_val, X_train = X_train_val[:5000] / 255.0, X_train_val[5000:] / 255.0
y_val, y_train = y_train_val[:5000], y_train_val[5000:]

model = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(300, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

model.save('./outputs/my_tf_model.keras')
model = load_model('./outputs/my_tf_model.keras')

# using callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

checkpoint_cb = ModelCheckpoint('./outputs/my_tf_model.keras')
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])

checkpoint_cb = ModelCheckpoint('./outputs/my_tf_model.keras', save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[checkpoint_cb])
model = tf.keras.models.load_model('./outputs/my_tf_model.keras')

early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb, early_stopping_cb])
model = load_model('./outputs/my_tf_model.keras')


class PrintValTrainRatioCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# using tensorboard for visualisation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.summary import create_file_writer

root_logdir = './logs'
print(root_logdir)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), callbacks=[tensorboard_cb])


test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1, 1000+1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
        data = (np.random.randn(100) + 2) * step / 100
        tf.summary.histogram("my_hist", data, buckets=50, step=step)
        images = np.random.randn(2, 32, 32, 3)
        tf.summary.image("my_images", images * step / 1000, step=step)
        texts = ["The step is ", str(step), "It's square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

# fine-tuning neural network hyperparameters
from tensorflow.keras.layers import InputLayer
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    optimiser = SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimiser)
    return model


tf_reg = KerasRegressor(build_model, n_hidden=3)
tf_reg.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=10)])
mse_test = tf_reg.score(X_test, y_test)

X_new = X_test[:3]
y_pred = tf_reg.predict(X_new)

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2)
}

tf_reg = KerasRegressor(build_fn=build_model)
rnd_search_cv = RandomizedSearchCV(tf_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train)
