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
import tensorflow as tf
import tensorflow.keras as tfk
import os
import time
import pandas as pd
import numpy as np
import math

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.cifar10.load_data()

X_train_scaled, X_val_scaled = X_train_val[:35000] / 255.0, X_train_val[35000:] / 255.0
y_train, y_val= y_train_val[:35000], y_train_val[35000:]
X_test_scaled = X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']

X_means = X_train_val[:35000].mean(axis=0)
X_stds = X_train_val[:35000].std(axis=0)

X_train_scaled2 = (X_train_val[:35000] - X_means) / X_stds


root_logdir = '.\logs'

def get_run_logdir(iteration="", learning_rate=""):
    import time
    extra_detail = iteration + learning_rate
    run_id = time.strftime("run-%Y_%m_%d-%H_%M_%S") + extra_detail
    return os.path.join(root_logdir, run_id)

# build a DNN with 20 hidden layers of 100 neurons each using he initialisation and the
# elu activation function
learning_rates = [1e-3, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 9e-5, 7e-5, 5e-5, 3e-5, 1e-5]
learning_rates = [5e-5, -1]

base_perf2 = []
for lr in learning_rates:
    tfk.backend.clear_session()
    model = tfk.models.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(tfk.layers.Dense(100, activation='elu', kernel_initializer='he_normal'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    # model.summary()

    # train the model using nadam optimisation and early stopping
    if lr < 0:
        print("Training a model with an exponentially decaying learning rate.")
        initial_lr = 1e-3
        final_lr = 1e-6
        num_epochs = 50
        batch_size = 32
        decay_factor = (final_lr / initial_lr)**(1 / num_epochs)
        steps = len(X_train_scaled) // batch_size
        lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr, decay_steps=steps, decay_rate=decay_factor,
            staircase=True)
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr_schedule)
        
        run_logdir = get_run_logdir("-base", "-exp_decay")
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        model_cb = tfk.callbacks.ModelCheckpoint("./outputs/cifar10_model_decay.keras",
                                                  save_best_only=True)
        callback_list = [tensorboard_cb, model_cb, earlystopping_cb]    
    else:
        print("Training a model with learning rate", str(lr))
        num_epochs = 50
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
        
        run_logdir = get_run_logdir("-base", "-" + str(lr))
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        model_cb = tfk.callbacks.ModelCheckpoint("./outputs/cifar10_model.keras",
                                                  save_best_only=True)
        callback_list = [tensorboard_cb, earlystopping_cb, model_cb]

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    print(run_logdir)
    
    t0 = time.time()
    history = model.fit(X_train_scaled, y_train, epochs=num_epochs,
                        callbacks=callback_list,
                        validation_data = (X_val_scaled, y_val))
    
    model_eval = model.evaluate(X_val_scaled, y_val)
    print(model_eval)
    run_time = time.time() - t0
    base_perf2.append(("base best", lr, model_eval[0], model_eval[1], run_time))

    print("Run time: ", run_time)

# the best model is the one with the lowest validation loss and achieved a validation
# accuracy of 0.4730. It had a had a learning rate of 5e-5, took 15.4 minutes to run and
# stopped after 23 epochs. The model with an exponentially decaying learning rate
# achieved a validation accuracy of 0.4792, took 17.4 minutes to run and stopped after
# 25 epochs.
base_perf = base_perf + base_perf2
base_perf_df = pd.DataFrame(base_perf, columns=["model", "learning_rate", "val_loss",
                                                "val_accuracy", "run_time"])
base_perf_df.to_csv("./outputs/base_perf.csv", index=False)


# add batch normalisation and compare the learning curves
learning_rates = [3e-3, 1e-3, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4]
learning_rates = [9e-4, -1]

bn_perf2 = []
for lr in learning_rates:
    tfk.backend.clear_session()
    model = tfk.models.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    model.add(tfk.layers.BatchNormalization())
    for _ in range(20):
        model.add(tfk.layers.Dense(100, kernel_initializer='he_normal'))
        model.add(tfk.layers.BatchNormalization())
        model.add(tfk.layers.Activation('elu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    # model.summary()

    if lr < 0:
        print("Training a model with an exponentially decaying learning rate.")
        initial_lr = 1e-3
        final_lr = 1e-6
        num_epochs = 50
        batch_size = 32
        decay_factor = (final_lr / initial_lr)**(1 / num_epochs)
        steps = len(X_train_scaled) // batch_size
        lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr, decay_steps=steps, decay_rate=decay_factor,
            staircase=True)
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr_schedule)
        
        run_logdir = get_run_logdir("-bn", "-exp_decay")
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        model_cb = tfk.callbacks.ModelCheckpoint("./outputs/bn_cifar10_model_decay.keras",
                                                  save_best_only=True)        
        callback_list = [tensorboard_cb, model_cb, earlystopping_cb]
    else:
        print("Training a model with learning rate", str(lr))
        num_epochs = 50
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
        
        run_logdir = get_run_logdir("-bn", "-" + str(lr))
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        model_cb = tfk.callbacks.ModelCheckpoint("./outputs/bn_cifar10_model.keras",
                                                  save_best_only=True)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10)
        callback_list = [tensorboard_cb, earlystopping_cb, model_cb]

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    print(run_logdir)
    
    t0 = time.time()   
    history = model.fit(X_train_scaled, y_train, epochs=num_epochs,
                        callbacks=callback_list,
                        validation_data = (X_val_scaled, y_val))

    model_eval = model.evaluate(X_val_scaled, y_val)
    print(model_eval)
    run_time = time.time() - t0
    bn_perf2.append(("bn", lr, model_eval[0], model_eval[1], run_time))

    print("Run time: ", run_time)
    
# the best model is the one with the lowest validation loss and achieved a validation
# accuracy of 0.5258. It had a had a learning rate of 9e-4, took 24.0 minutes to run and
# stopped after 29 epochs. The model with an exponentially decaying learning rate
# achieved a validation accuracy of 0.5273, took 23.9 minutes to run and stopped after
# 27 epochs.
bn_perf = bn_perf + bn_perf2
bn_perf_df = pd.DataFrame(bn_perf, columns=["model", "learning_rate", "val_loss",
                                            "val_accuracy", "run_time"])
bn_perf_df.to_csv("./outputs/bn_perf.csv", index=False)


# replace batch normalisation with selu and make the necessary adjustments to ensure the
# DNN self-normalises (e.g. standardise input features, use LeCun normal initialisation,
# use only a sequence of dense layers)
X_means = X_train_scaled.mean(axis=0)
X_stds = X_train_scaled.std(axis=0)

X_train_std = (X_train_scaled - X_means) / X_stds
X_val_std = (X_val_scaled - X_means) / X_stds
X_test_std = (X_test_scaled - X_means) / X_stds

learning_rates = [1e-3, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4]
learning_rates = [1e-3, -1]

selu_perf2 = []
for lr in learning_rates:
    tfk.backend.clear_session()
    model = tfk.models.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(tfk.layers.Dense(100, activation='selu',
                                   kernel_initializer='lecun_normal'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    # model.summary()

    if lr < 0:
        print("Training a model with an exponentially decaying learning rate.")
        initial_lr = 1e-3
        final_lr = 1e-6
        num_epochs = 50
        batch_size = 32
        decay_factor = (final_lr / initial_lr)**(1 / num_epochs)
        steps = len(X_train_scaled) // batch_size
        lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr, decay_steps=steps, decay_rate=decay_factor,
            staircase=True)
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr_schedule)
        
        run_logdir = get_run_logdir("-selu", "-exp_decay")
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        model_cb = tfk.callbacks.ModelCheckpoint(
            "./outputs/selu_cifar10_model_decay.keras", save_best_only=True)        
        callback_list = [tensorboard_cb, model_cb, earlystopping_cb]
    else:
        print("Training a model with learning rate", str(lr))
        num_epochs = 50
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
        
        run_logdir = get_run_logdir("-selu", "-" + str(lr))
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        model_cb = tfk.callbacks.ModelCheckpoint("./outputs/selu_cifar10_model.keras",
                                                  save_best_only=True)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        callback_list = [tensorboard_cb, earlystopping_cb, model_cb]

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    print(run_logdir)
    
    t0 = time.time()   
    history = model.fit(X_train_std, y_train, epochs=num_epochs,
                        callbacks=callback_list,
                        validation_data = (X_val_std, y_val))

    model_eval = model.evaluate(X_val_std, y_val)
    print(model_eval)
    run_time = time.time() - t0
    selu_perf2.append(("selu", lr, model_eval[0], model_eval[1], run_time))

    print("Run time: ", run_time)

# the best model is the one with the lowest validation loss and achieved a validation
# accuracy of 0.4837. It had a had a learning rate of 1e-3, took 12.9 minutes to run and
# stopped after 24 epochs. The model with an exponentially decaying learning rate
# achieved a validation accuracy of 0.4901, took 9.5 minutes to run and stopped after
# 17 epochs.
selu_perf = selu_perf + selu_perf2
selu_perf_df = pd.DataFrame(selu_perf, columns=["model", "learning_rate", "val_loss",
                                            "val_accuracy", "run_time"])
selu_perf_df.to_csv("./outputs/selu_perf.csv", index=False)
    
    
# regularise the model with alpha dropout
learning_rates = [1e-3, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 9e-5, 7e-5, 5e-5, 3e-5, 1e-5]
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
learning_rates = [9e-4, -1]
dropout_rates = [0.1]

drop_perf2 = []
for lr in learning_rates:
    for dr in dropout_rates:
        tfk.backend.clear_session()
        model = tfk.models.Sequential()
        model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
        for _ in range(20):
            model.add(tfk.layers.Dense(100, activation='selu',
                                       kernel_initializer='lecun_normal'))
        model.add(tfk.layers.AlphaDropout(dr))
        model.add(tfk.layers.Dense(10, activation='softmax'))
        # model.summary()
    
        # train the model using nadam optimisation and early stopping
        if lr < 0:
            print("Training a model with an exponentially decaying learning rate.")
            initial_lr = 1e-3
            final_lr = 1e-6
            num_epochs = 50
            batch_size = 32
            decay_factor = (final_lr / initial_lr)**(1 / num_epochs)
            steps = len(X_train_scaled) // batch_size
            lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_lr, decay_steps=steps,
                decay_rate=decay_factor, staircase=True)
            optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr_schedule)
            
            run_logdir = get_run_logdir("-drop" + str(dr), "-exp_decay")
            tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
            earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                           restore_best_weights=True)
            model_cb = tfk.callbacks.ModelCheckpoint(
                "./outputs/drop_cifar10_model_decay.keras", save_best_only=True)        
            callback_list = [tensorboard_cb, earlystopping_cb, model_cb]
        else:
            print("Training a model with learning rate", str(lr))
            num_epochs = 50
            optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
            
            run_logdir = get_run_logdir("-drop" + str(dr), "-" + str(lr))
            tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
            model_cb = tfk.callbacks.ModelCheckpoint(
                "./outputs/drop_cifar10_model.keras", save_best_only=True)
            earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                           restore_best_weights=True)
            callback_list = [earlystopping_cb, tensorboard_cb, model_cb]
    
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                      metrics=['accuracy'])
        print(run_logdir)
        
        t0 = time.time()   
        history = model.fit(X_train_std, y_train, epochs=num_epochs,
                            callbacks=callback_list,
                            validation_data = (X_val_std, y_val))
    
        model_eval = model.evaluate(X_val_std, y_val)
        print(model_eval)
        run_time = time.time() - t0
        drop_perf.append(("drop", lr, dr, model_eval[0], model_eval[1], run_time))
    
        print("Run time: ", run_time)

# the best model is the one with the lowest validation loss and achieved a validation
# accuracy of 0.4499. It had a had a learning rate of 9e-4, took 3.16 minutes to run and
# stopped after 22 epochs. The model with an exponentially decaying learning rate
# achieved a validation accuracy of 0.4893, took 2.5 minutes to run and stopped after
# 18 epochs.
drop_perf = drop_perf + drop_perf2
drop_perf_df = pd.DataFrame(drop_perf, columns=["model", "learning_rate", "dropout rate",
                                                "val_loss", "val_accuracy", "run_time"])
drop_perf_df.to_csv("./outputs/drop_perf.csv", index=False)


# without retraining the best dropout model, see if MC Dropout increases accuracy
tensorflow.random.set_seed(42)
y_probas = np.stack([model(X_test_std, training=True) for sample in range(100)])
model.predict(X_test_std[:1]).round(3)

y_std = y_probas.std(axis=0)
y_std[0].round(3)

y_pred = np.argmax(y_proba, axis=1)
MC_accuracy = (y_pred == y_test.reshape(-1,)).sum() / len(y_test)
print(MC_accuracy)

print(model.evaluate(X_test_std, y_test))
# accuracy on the test set for the model with the exponentially decaying learning rate
# was 0.4837. With MCDropout, accuracy dropped marginally to 0.4833.


# retrain the model/s with 1cycle scheduling and see if it improves training speed and
# accuracy
class OneCycle(tfk.optimizers.schedules.LearningRateSchedule):
    """A OneCycle that uses an cosine annealing schedule for cycle."""

    def __init__(
        self,
        initial_learning_rate,
        maximal_learning_rate,
        cycle_size,
        scale_fn=lambda x: 1.0,
        shift_peak=0.3,
        scale_mode="cycle",
        final_lr_scale=1.0,
        name=None,
    ):
        """
        Applies cyclical cosine annealing learning rate.
        It is possible to get the same learning rate scheduler as it was
        used by FastAI for superconvergence: https://docs.fast.ai/callbacks.one_cycle.html
        or Kaggle post: https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy
        In order to do that:
        ```python
         maximal_learning_rate = <value from lr finder>
        initial_learning_rate = maximal_learning_rate / 25.0
        cycle_size = 3-5 epoch (should be defined by you) It defines size of cycle
        lr_schedule = CyclicalCosineAnnealing(
            initial_learning_rate,
            maximal_learning_rate,
            cycle_size,
            scale_fn = lambda x: 1.0, or lambda x: tf.where(x > 1.0, 0.8, 1.0)
            shift_peak = 0.3
            scale_mode="cycle",
           final_lr_scale=1e-4)
        ```
        The learning rate schedule is also serializable and deserializable using
        `tf.keras.optimizers.schedules.serialize` and
        `tf.keras.optimizers.schedules.deserialize`.
        Args:
        initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The maximal learning rate.
        step_size: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the  half cycle size in interations.
        scale_fn: scale your cycle (make it bigger/smaller for the next cycle)
        shift_peak: shift the pick point to the left side
        scale_mode: scale by "cycle" or "step"
        final_lr_scale: filal_lr = initial_learning_rate * final_lr_scale
        name: String.  Optional name of the operation.
        Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the cyclical learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
        """
        super(OneCycle, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.cycle_size = cycle_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.shift_peak = shift_peak
        self.final_lr_scale = final_lr_scale
        self.name = name
        # Defines the position of the max lr in steps
        self._total_steps = cycle_size
        self._first_half_steps = shift_peak * self._total_steps
        self._second_half_steps = self._total_steps - self._first_half_steps

    def get_cosine_annealing(self, start, end, step, step_size_part, cycle):
        x = step / step_size_part
        cosine_annealing = 1 + tf.math.cos(tf.constant(np.pi) * x)
        return end + 0.5 * (start - end) * cosine_annealing

    def __call__(self, step, optimizer=False):
        with tf.name_scope(self.name or "OneCycle"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            # Cast all internal members to necessary type
            step = tf.cast(step, dtype)
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            first_half_steps = tf.cast(self._first_half_steps, dtype)
            second_half_steps = tf.cast(self._second_half_steps, dtype)
            total_steps = tf.cast(self._total_steps, dtype)
            final_lr_scale = tf.cast(self.final_lr_scale, dtype)
            # Check in % the cycle
            cycle_progress = step / total_steps
            cycle = tf.floor(1 + cycle_progress)

            percentage_complete = 1.0 - tf.abs(cycle - cycle_progress)  # percent of iterations done
            first_half = tf.cast(percentage_complete <= self.shift_peak, dtype)

            normalized_first_half_step = step - (cycle - 1) * total_steps
            normalized_second_half_step = normalized_first_half_step - first_half_steps
            final_lr = initial_learning_rate * final_lr_scale

            lr_begin = self.get_cosine_annealing(
                initial_learning_rate,
                maximal_learning_rate,
                normalized_first_half_step,
                first_half_steps,
                cycle,
            )
            lr_end = self.get_cosine_annealing(
                maximal_learning_rate,
                final_lr,
                normalized_second_half_step,
                second_half_steps,
                cycle,
            )

            lr_res = first_half * lr_begin + (1.0 - first_half) * lr_end
            mode_step = cycle if self.scale_mode == "cycle" else step

            if optimizer == False:
                lr_res = lr_res * self.scale_fn(mode_step)

            return lr_res

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "cycle_size": self.cycle_size,
            "scale_mode": self.scale_mode,
            "shift_peak": self.shift_peak,
        }

tfk.backend.clear_session()
model = tfk.models.Sequential()
model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tfk.layers.Dense(100, activation='elu',
                               kernel_initializer='he_normal'))
model.add(tfk.layers.Dense(10, activation='softmax'))
# model.summary()

run_logdir = get_run_logdir("-base", "-" + str(lr))
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
print(run_logdir)
earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10)

maximal_learning_rate = 1e-3
initial_learning_rate = maximal_learning_rate / 25.0
batch_size = 32
num_steps_per_epoch = len(X_train_scaled) // batch_size
num_epochs = 50
cycle_size = num_epochs / num_steps_per_epoch
lr_schedule = OneCycle(
    initial_learning_rate=initial_learning_rate,
    maximal_learning_rate=maximal_learning_rate,
    cycle_size=cycle_size,
    scale_mode='cycle',
    shift_peak=0.2,
    name='CyclicalCosine',
)

optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr_schedule)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
              metrics=['accuracy'])
        
t0 = time.time()
history = model.fit(X_train_scaled, y_train, epochs=num_epochs,
                    callbacks=[tensorboard_cb, earlystopping_cb],
                    validation_data = (X_val_scaled, y_val))

model_eval = model.evaluate(X_val_scaled, y_val)
print(model_eval)
run_time = time.time() - t0
print("Run time: ", run_time)
