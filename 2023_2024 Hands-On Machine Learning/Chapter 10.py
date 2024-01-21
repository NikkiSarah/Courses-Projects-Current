#%% from biological to artifical neurons

# the perceptron
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)

clf = Perceptron(random_state=42)
clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = clf.predict(X_new)

# regression MLPs
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target,
                                                            random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(rmse)

#%% implementing MLPs with keras
# building an image classifier using the sequential API
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tfk.datasets.fashion_mnist.load_data()
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_val[:-5000], y_train_val[:-5000]
X_val, y_val = X_train_val[-5000:], y_train_val[-5000:]
print(X_train.shape)
print(X_train.dtype)

X_train, X_val, X_test = X_train / 255., X_val / 255., X_test / 255.
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']
print(class_names[y_train[0]])

model = tfk.models.Sequential()
model.add(tfk.layers.Input(shape=[28, 28]))
model.add(tfk.layers.Flatten())
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

model.summary()
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

pd.DataFrame(history.history).plot(
    xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=-1)
print(y_pred)
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(y_new)

# building a regression MLP using the sequential API
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tfk

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target,
                                                            random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  random_state=42)

tf.random.set_seed(42)
norm_layer = tfk.layers.Normalization(input_shape=X_train.shape[1:])
model = tfk.Sequential([
    norm_layer,
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(1)
    ])
optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimiser, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

# building complex models using the functional API
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tfk

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(housing.data, housing.target,
                                                            random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  random_state=42)

normalisation_layer = tfk.layers.Normalization()
hidden_layer1 = tfk.layers.Dense(30, activation='relu')
hidden_layer2 = tfk.layers.Dense(30, activation='relu')
concat_layer = tfk.layers.Concatenate()
output_layer = tfk.layers.Dense(1)

input_ = tfk.layers.Input(shape=X_train.shape[1:])
normalised = normalisation_layer(input_)
hidden1 = hidden_layer1(normalised)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalised, hidden2])
output = output_layer(concat)

model = tfk.Model(inputs=[input_], outputs=[output])


tfk.backend.clear_session()
input_wide = tfk.layers.Input(shape=[5])
input_deep = tfk.layers.Input(shape=[6])
norm_layer_wide = tfk.layers.Normalization()
norm_layer_deep = tfk.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tfk.layers.Dense(30, activation='relu')(norm_deep)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.concatenate([norm_wide, hidden2])
output = tfk.layers.Dense(1)(concat)
model = tfk.Model(inputs=[input_wide, input_deep], outputs=[output])

optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimiser, metrics=['RootMeanSquaredError'])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_val_wide, X_val_deep = X_val[:, :5], X_val[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
                    validation_data=((X_val_wide, X_val_deep), y_val))
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
print(y_pred)


tfk.backend.clear_session()
input_wide = tfk.layers.Input(shape=[5])
input_deep = tfk.layers.Input(shape=[6])
norm_layer_wide = tfk.layers.Normalization()
norm_layer_deep = tfk.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tfk.layers.Dense(30, activation='relu')(norm_deep)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.concatenate([norm_wide, hidden2])
output = tfk.layers.Dense(1)(concat)
aux_output = tfk.layers.Dense(1)(hidden2)
model = tfk.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])

optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=('mse', 'mse'), loss_weights=(0.9, 0.1), optimizer=optimiser,
              metrics=['RootMeanSquaredError'])

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
    validation_data=((X_val_wide, X_val_deep), (y_val, y_val))
    )
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results

y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
y_pred_tuple = model.predict((X_new_wide, X_new_deep))
y_pred = dict(zip(model.output_names, y_pred_tuple))
print(y_pred)

# using the subclassing API to build dynamic models
import tensorflow.keras as tfk

class WideAndDeepModel(tfk.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.norm_layer_wide = tfk.layers.Normalization()
        self.norm_layer_deep = tfk.layers.Normalization()
        self.hidden1 = tfk.layers.Dense(units, activation=activation)
        self.hidden2 = tfk.layers.Dense(units, activation=activation)
        self.main_output = tfk.layers.Dense(1)
        self.aux_output = tfk.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tfk.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

model = WideAndDeepModel(30, activation='relu', name='my_cool_model')

optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=optimiser,
              metrics=["RootMeanSquaredError"])
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_val_wide, X_val_deep), (y_val, y_val)))
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# saving and restoring a model
model.save('./outputs/my_keras_model', save_format='tf')

model = tfk.models.load_model('my_keras_model')
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))


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
checkpoint_cb = tfk.callbacks.ModelCheckpoint('./outputs/my_tf_model.keras')
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])

checkpoint_cb = tfk.callbacks.ModelCheckpoint('./outputs/my_tf_model.keras',
                                          save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb])
model = tfk.saving.load_model('./outputs/my_tf_model.keras')

early_stopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
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
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
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
           callbacks=[tfk.callbacks.EarlyStopping(patience=10)])
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
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
early_stopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
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
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
early_stopping_cb = tfk.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint_cb = tfk.callbacks.ModelCheckpoint('./outputs/best_tf_model.keras',
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
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
early_stopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint_cb = tfk.callbacks.ModelCheckpoint('./outputs/final_tf_model.keras',
                                                save_best_only=True)
history = model.fit(X_train, y_train_ohe, epochs=30, validation_data=(X_val, y_val_ohe),
                    callbacks=[tensorboard_cb, early_stopping_cb, model_checkpoint_cb])

final_model = tfk.saving.load_model('./outputs/final_tf_model.keras')
y_pred = np.argmax(final_model.predict(X_test), axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(test_accuracy)

test_precision = precision_score(y_test, y_pred, average='micro')
print(test_precision)
