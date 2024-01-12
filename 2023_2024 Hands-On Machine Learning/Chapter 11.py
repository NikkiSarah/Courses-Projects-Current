# %% the vanishing/exploding gradients problem
import tensorflow
import tensorflow.keras as tfk

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tfk.datasets.fashion_mnist
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()

X_val_scaled, X_train_scaled = X_train_val[:5000] / 255.0, X_train_val[5000:] / 255.0
y_val, y_train = y_train_val[:5000], y_train_val[5000:]
X_test_scaled = X_test / 255.0
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']

# batch normalisation
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(10, activation='softmax')
])
model.summary()

[(var.name, var.trainable) for var in model.layers[1].variables]

tfk.backend.clear_session()
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    tfk.layers.BatchNormalization(),
    tfk.layers.Activation('elu'),
    tfk.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    tfk.layers.BatchNormalization(),
    tfk.layers.Activation('elu'),
    tfk.layers.Dense(10, activation='softmax')
])
model.summary()

# gradient clipping
optimiser = tfk.optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimiser)

# %% reusing pretrained layers

# transfer learning
model_A = tfk.saving.load_model('./outputs/my_model_A.keras')
model_B_on_A = tfk.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(tfk.layers.Dense(1, activation='sigmoid'))
model_A_clone = tfk.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_val_B, y_val_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimiser = tfk.optimizers.SGD(learning_rate=1e-4)
model_B_on_A.compile(loss='binary_crossentropy', optimizer=optimiser,
                     metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_val_B, y_val_B))

model_B_on_A.evaluate(X_test_B, y_test_B)

# %% faster optimisers

# momentum
optimiser = tfk.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
# nesterov accelerated gradient
optimiser = tfk.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
# rmsprop
optimiser = tfk.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
# adam and nadam
optimiser = tfk.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)

# learning rate scheduling

# power
optimiser = tfk.optimizers.legacy.SGD(learning_rate='SGD', decay=1e-4)


# exponential and piecewise
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

lr_scheduler = tfk.callbacks.LearningRateScheduler(exponential_decay_fn)
model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD',
              metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=20,
                    validation_data = (X_val_scaled, y_val), callbacks=[lr_scheduler])


def exponential_decay_fn(epoch, lr):
    return lr * 0.1**(1 / 20)


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


# performance
lr_scheduler = tfk.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

s = 20 * len(X_train) // 32
learning_rate = tfk.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimiser = tfk.optimizers.SGD(learning_rate)

# %% avoiding overfitting through regularisation
from functools import partial
import numpy as np

# l1 and l2 regularisation
layer = tfk.layers.Dense(100, activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=tfk.regularizers.L2(0.01))

RegularisedDense = partial(tfk.layers.Dense, activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=tfk.regularizers.L2(0.01))
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    RegularisedDense(300),
    RegularisedDense(100),
    RegularisedDense(10, activation='softmax', kernel_initializer='glorot_uniform')
    ])
                         
# dropout
tfk.backend.clear_session()
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD',
              metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=20,
                    validation_data = (X_val_scaled, y_val))

# monte carlo dropout
y_probas = np.stack([model(X_test_scaled, training=True) for sample in range(100)])
y_proba = y_probas.mean(axis=1)

print(np.round(model.predict(X_test_scaled[:1]), 2))
print(np.round(y_probas[:3, :1], 2))
print(np.round(y_probas[:1], 2))

y_std = y_probas.std(axis=0)
print(np.round(y_std[:1], 2))

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(accuracy)


class MCDropout(tfk.layers.Dropout):
    def call(self, input):
        return super().call(inputs, training=True)


# max-norm regularisation
tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal',
                 kernel_constraint=tfk.constraints.MaxNorm(1.))


# %% coding exercises

# build a DNN on the CIFAR10 image dataset
import tensorflow
import tensorflow.keras as tfk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.cifar10.load_data()

X_val_scaled, X_train_scaled = X_train_val[:40000] / 255.0, X_train_val[40000:] / 255.0
y_val, y_train = y_train_val[:40000], y_train_val[40000:]
X_test_scaled = X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']

# build a DNN with 20 hidden layers of 100 neurons each using he initialisation and the
# elu activation function
# batch normalisation
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[32, 32, 3]),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tfk.layers.Dense(10, activation='softmax')
])
model.summary()

# train the model using nadam optimisation and early stopping. Remember to search for the
# correct learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
              metrics=['accuracy'])
cb_earlystopping = tfk.callbacks.EarlyStopping(patience=10)
history = model.fit(X_train_scaled, y_train, epochs=20, callbacks=[cb_earlystopping],
                    validation_data = (X_val_scaled, y_val))

# add batch normalisation and compare the learning curves

# replace batch normalisation with selu and make the necessary adjustments to ensure the
# DNN self-normalises (e.g. standardise input features, use LeCun normal initialisation,
# use only a sequence of dense layers)

# regularise the model with alpha dropout

# without retraining the model, see if MC Dropout results in better accuracy

# retrain the model with 1cycle scheduling and see if it improves training speed and
# accuracy
