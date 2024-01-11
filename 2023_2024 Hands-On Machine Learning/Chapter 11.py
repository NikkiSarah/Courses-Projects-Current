# %% batch normalisation
import tensorflow
from tensorflow.keras import *

model = models.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.BatchNormalization(),
    layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])
model.summary()

print([(var.name, var.trainable) for var in model.layers[1].variables])

backend.clear_session()
model = models.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.BatchNormalization(),
    layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dense(10, activation='softmax')
])
model.summary()

# %% gradient clipping
optimiser = optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimiser)

