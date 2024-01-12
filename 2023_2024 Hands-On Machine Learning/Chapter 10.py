# %% from biological to artifical neurons

# the perceptron
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(int)

clf = Perceptron()
clf.fit(X, y)
y_pred = clf.predict([[2, 0.5]])

# %% implementing MLPs with keras
# installing tensorflow
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

print(tf.__version__)

# building an image classifier using the sequential API
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tfk.datasets.fashion_mnist
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_val.shape)
print(X_train_val.dtype)

X_val, X_train = X_train_val[:5000] / 255.0, X_train_val[5000:] / 255.0
y_val, y_train = y_train_val[:5000], y_train_val[5000:]
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']
print(class_names[y_train[0]])

model = tfk.models.Sequential()
model.add(tfk.layers.Flatten(input_shape=[28, 28]))
model.add(tfk.layers.Dense(300, activation='relu'))
model.add(tfk.layers.Dense(100, activation='relu'))
model.add(tfk.layers.Dense(10, activation='softmax'))

tfk.backend.clear_session()
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])

print(model.summary())
model.layers
hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD',
              metrics=['accuracy'])
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

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = tfk.models.Sequential([
    tfk.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    tfk.layers.Dense(1)
    ])
model.compile(loss='mean_squared_error', optimizer='SGD')
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
mse_test = model.evaluate(X_test, y_test)
print(mse_test)
X_new = X_test[:3]
print(X_new)
y_pred = model.predict(X_new)
print(y_pred)

# building complex models using the functional API
input_ = tfk.Input(shape=X_train.shape[1:])
hidden1 = tfk.layers.Dense(30, activation='relu')(input_)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.Concatenate()([input_, hidden2])
output = tfk.layers.Dense(1)(concat)
model = tfk.Model(inputs=[input_], outputs=[output])

input_A = tfk.Input(shape=[5], name="wide_input")
input_B = tfk.Input(shape=[6], name="deep_input")
hidden1 = tfk.layers.Dense(30, activation='relu')(input_B)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.Concatenate()([input_A, hidden2])
output = tfk.layers.Dense(1, name="output")(concat)
model = tfk.Model(inputs=[input_A, input_B], outputs=[output])
model.compile(loss='mse', optimizer=tfk.optimizers.SGD(learning_rate=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_val_A, X_val_B = X_val[:, :5], X_val[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_val_A, X_val_B), y_val))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
print(mse_test)
y_pred = model.predict((X_new_A, X_new_B))
print(y_pred)

output = tfk.layers.Dense(1, name="main_output")(concat)
aux_output = tfk.layers.Dense(1, name="aux_output")(hidden2)
model = tfk.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='SGD')

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_val_A, X_val_B], [y_val, y_val]))
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
print(y_pred_main)
print(y_pred_aux)

# using the subclassing API to build dynamic models
class WideAndDeepModel(tfk.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tfk.layers.Dense(units, activation=activation)
        self.hidden2 = tfk.layers.Dense(units, activation=activation)
        self.main_output = tfk.layers.Dense(1)
        self.aux_output = tfk.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = tfk.layers.Concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


model = WideAndDeepModel()

# saving and restoring a model
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()
X_val, X_train = X_train_val[:5000] / 255.0, X_train_val[5000:] / 255.0
y_val, y_train = y_train_val[:5000], y_train_val[5000:]

model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

model.save('./outputs/my_tf_model.keras')
model = tfk.saving.load_model('./outputs/my_tf_model.keras')

# using callbacks
checkpoint_cb = tfk.callback.ModelCheckpoint('./outputs/my_tf_model.keras')
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])

checkpoint_cb = tfk.callback.ModelCheckpoint('./outputs/my_tf_model.keras',
                                          save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb])
model = tfk.saving.load_model('./outputs/my_tf_model.keras')

early_stopping_cb = tfk.callback.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb, early_stopping_cb])
model = tfk.saving.load_model('./outputs/my_tf_model.keras')


class PrintValTrainRatioCallback(tfk.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# %% using tensorboard for visualisation
import os
import tensorflow.summary as tfs

root_logdir = './logs'
print(root_logdir)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = tfk.callback.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val),
                    callbacks=[tensorboard_cb])


test_logdir = get_run_logdir()
writer = tfs.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1, 1000+1):
        tfs.scalar("my_scalar", np.sin(step / 10), step=step)
        data = (np.random.randn(100) + 2) * step / 100
        tfs.histogram("my_hist", data, buckets=50, step=step)
        images = np.random.randn(2, 32, 32, 3)
        tfs.image("my_images", images * step / 1000, step=step)
        texts = ["The step is ", str(step), "It's square is " + str(step**2)]
        tfs.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tfs.audio("my_audio", audio, sample_rate=48000, step=step)

# %% fine-tuning neural network hyperparameters
from scikeras.wrappers import KerasRegressor
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = tfk.models.Sequential()
    model.add(tfk.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(tfk.layers.Dense(n_neurons, activation='relu'))
    model.add(tfk.layers.Dense(1))
    optimiser = tfk.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimiser)
    return model


tf_reg = KerasRegressor(build_model, n_hidden=3)
tf_reg.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
           callbacks=[tfk.callback.EarlyStopping(patience=10)])
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
rnd_search_cv.best_params_
rnd_search_cv.best_score_

model = rnd_search_cv.best_estimator_.model

# %% coding exercises
# train a deep MLP
# import ssl
from sklearn.metrics import accuracy_score, precision_score

# ssl._create_default_https_context = ssl._create_unverified_context
fashion_mnist = tfk.datasets.fashion_mnist
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()

X_train_val = X_train_val.reshape((60000, 28, 28, 1)) / 255.0
X_val, X_train = X_train_val[:5000], X_train_val[5000:]
y_val, y_train = y_train_val[:5000], y_train_val[5000:]

y_train_ohe = tfk.utils.to_categorical(y_train)
y_val_ohe = tfk.utils.to_categorical(y_val)

model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='SGD', loss='categorical_crossentropy',
              metrics=['accuracy', tfk.metrics.Precision()])

run_logdir = get_run_logdir()
tensorboard_cb = tfk.callback.TensorBoard(run_logdir)
early_stopping_cb = tfk.callback.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train_ohe, epochs=100, validation_data=(X_val, y_val_ohe),
                    callbacks=[tensorboard_cb, early_stopping_cb])

y_pred = np.argmax(model.predict(X_val), axis=1)
val_accuracy = accuracy_score(y_val, y_pred)
print(val_accuracy)

val_precision = precision_score(y_val, y_pred, average='micro')
print(val_precision)


# determine the optimal learning rate using an exponential increase
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])

tfk.backend.clear_session()

def step_decay(epoch, lr):
    if epoch < 1:
        new_lr = lr
        print(new_lr)
        return lr
    else:
        new_lr = lr * tf.math.exp(1.005)
        print(new_lr)
        return new_lr


lr_schedule_cb = tfk.optimizers.schedules.LearningRateScheduler(step_decay)
model.compile(optimizer=tfk.optimizers.SGD(1e-5), loss='categorical_crossentropy',
              metrics=['accuracy', tfk.metrics.Precision()])

run_logdir = get_run_logdir()
tensorboard_cb = tfk.callback.TensorBoard(run_logdir)
early_stopping_cb = tfk.callback.EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint_cb = tfk.callback.ModelCheckpoint('./outputs/best_tf_model.keras',
                                                save_best_only=True)
history = model.fit(X_train, y_train_ohe, epochs=30, validation_data=(X_val, y_val_ohe),
                    callbacks=[tensorboard_cb, lr_schedule_cb, early_stopping_cb,
                               model_checkpoint_cb])

best_model = tfk.saving.load_model('./outputs/best_tf_model.keras')
optimiser = best_model.optimizer
best_learning_rate = optimiser.learning_rate.numpy()
print(best_learning_rate)

model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])

tfk.backend.clear_session()
model.compile(optimizer=tfk.optimizers.SGD(learning_rate=best_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', tfk.metrics.Precision()])

run_logdir = get_run_logdir()
tensorboard_cb = tfk.callback.TensorBoard(run_logdir)
early_stopping_cb = tfk.callback.EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint_cb = tfk.callback.ModelCheckpoint('./outputs/final_tf_model.keras',
                                                save_best_only=True)
history = model.fit(X_train, y_train_ohe, epochs=30, validation_data=(X_val, y_val_ohe),
                    callbacks=[tensorboard_cb, early_stopping_cb, model_checkpoint_cb])

final_model = tfk.saving.load_model('./outputs/final_tf_model.keras')
y_pred = np.argmax(final_model.predict(X_test), axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(test_accuracy)

test_precision = precision_score(y_test, y_pred, average='micro')
print(test_precision)
