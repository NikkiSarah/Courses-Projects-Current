import tensorflow as tf
import numpy as np
import tensorflow.keras as tfk

# %% using tensorflow like numpy

# tensors and operations
tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(42))

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)
print(t.dtype)

print(t[:, 1:])
print(t[..., 1, tf.newaxis])

print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))

# tensors and numpy
a = np.array([2., 4., 5.])
print(tf.constant(a))
print(np.array(t))
print(tf.square(a))
print(np.square(t))

# type conversions
print(tf.constant(2.) + tf.constant(40))
print(tf.constant(2.) + tf.constant(40, dtype=np.float64))

t2 = tf.constant(40, dtype=tf.float64)
print(tf.constant(2.0) + tf.cast(t2, tf.float32))

# variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)

print(v.assign(2 * v))
print(v[0, 1].assign(42))
print(v[:, 2].assign([0., 1.]))
print(v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.]))

#%% customising models and training algorithms

# custom loss functions
# Huber is now a defined loss function in tensorflow: from tfk.losses.Huber /
# loss='huber_loss'
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer='Nadam', metrics='mae')
model.fit(X_train, y_train, [...])

# saving and loading models with custom components
model = tfk.models.load_model("my_model_with_a_custom_loss_threshold_2.keras",
                              custom_objects={"huber_fn": create_huber(2.0)})

class HuberLoss(tfk.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


model.compile(loss=HuberLoss(2.), optimizer="nadam")
model = tfk.models.load_model("my_model_with_a_custom_loss_class.keras",
                              custom_objects={"HuberLoss": HuberLoss})

# custom activation functions, initialisers, regularisers and constraints
def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initaliser(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regulariser(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = tfk.layers.Dense(30, activation=my_softplus,
                         kernel_initializer=my_glorot_initaliser,
                         kernel_regularizer=my_l1_regulariser,
                         kernel_constraint=my_positive_weights)


class MyL1Regulariser(tfk.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor": self.factor}

# custom metrics
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])

precision = tfk.metrics.Precision()
print(precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1]))
print(precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0]))

print(precision.result())        
print(precision.variables)
precision.reset_states()

class HuberMetric(tfk.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total_assign_add = tf.reduce_sum(metric)
        self.count_assign_add(tf.cast(tf.size(y_true), tf.float32))
        
    def result(self):
        return self.total / self.count
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
    
# custom layers
exponential_layer = tfk.layers.Lambda(lambda x: tf.exp(x))

class MyDense(tfk.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tfk.activations.get(activation)
        
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel",
                                      shape=[batch_input_shape[-1], self.units],
                                      initializer="he_normal")
        self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)
        
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tfk.activations.serialize(self.activation)}
 
    
class MyMultiLayer(tfk.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return [X1 + X2, X1 * X2, X1 / X2]
    
    def compute_output_shape(self, batch_input_shape):
        b1, b2 = batch_input_shape
        return [b1, b1, b1]


class MyGaussianNoise(tfk.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X
    
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


# custom models
class ResidualBlock(tfk.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tfk.layers.Dense(n_neurons, activation="relu",
                                        kernel_initializer="he_normal")
                       for _ in range(n_layers)]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
            return inputs + Z


class ResidualRegressor(tfk.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tfk.layers.Dense(30, activation="relu", 
                                        kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tfk.layers.Dense(output_dim)
    
    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z - self.block2(Z)
        return self.out(Z)


# losses and metrics based on model internals
class ReconstructingRegressor(tfk.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tfk.layers.Dense(30, activation="selu",
                                        kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = tfk.layers.Dense(output_dim)
        self.reconstruction_mean = tfk.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tfk.layers.Dense(n_inputs)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)


# computing gradients using autodiff
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2


w1, w2 = 5, 3
eps = 1e-6
print((f(w1 + eps, w2) - f(w1, w2)) / eps)
print((f(w1, w2 + eps) - f(w1, w2)) / eps)

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
print(gradients)

with tf.GradientTape() as tape:
    z = f(w1, w2)
dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)

with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
dz_dq1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)
del tape

c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])

with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])

def f(w1, w2):
    return 3 * w1**2 + tf.stop_gradient(2 * w1 * w2)
with tf.GradientTape() as tape:
    z = f(w1, w2)
gradients = tape.gradient(z, [w1, w2])

x = tf.Variable([100.])
with tf.GradientTape() as tape:
    z = my_softplus(x)
tape.gradient(z, [x])

@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)
    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), my_softplus_gradients
    
# custom training loops
l2_reg = tfk.regularizers.l2(0.05)
model = tfk.models.Sequential([
    tfk.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                     kernel_regularizer=l2_reg),
    tfk.layers.Dense(1, kernel_regularizer=l2_reg)
    ])    

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print(f"\r{iteration}/{total} - " + metrics, end=end)

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimiser = tfk.optimizers.Nadam(learning_rate=0.01)
loss_fn = tfk.losses.mean_squared_error
mean_loss = tfk.metrics.Mean()
metrics = [tfk.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model_losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
        for metric in [mean_loss] + metrics:
            metric.reset_states()
    
# %% tensorflow functions and graphs

def cube(x):
    return x**3

print(cube(2))
print(cube(tf.constant(2.0)))

tf_cube = tf.function(cube)
print(tf_cube)

print(tf_cube(2))
print(tf_cube(tf.constant(2.0)))

@tf.function
def tf_cube(X):
    return x**3

print(tf_cube.python_function(2))

# %% coding exercises
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import numpy as np

# implement a custom layer that performs layer normalisation
class LayerNormalisation(tfk.layers.Layer):
    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

# - the build() method should define two trainable weights alpha and beta, both of shape
#   input_shape[-1:] and data type tf.float32. Alpha should be initialised with 1s and
#   beta with zeros
    def build(self, batch_input_shape):
        self.alpha = self.add_weight(name="alpha", shape=batch_input_shape[-1:],
                                     dtype=np.float32, initializer="ones")
        self.beta = self.add_weight(name="beta", shape=batch_input_shape[-1:],
                                    dtype=np.float32, initializer="zeros")

# - the call() method should compute the mean and standard devision of each instance's
#   features. You can use tf.nn.moments(inputs, axes=-1, keepdims=True), which returns
#   the mean and variance of all instances. Then the function should compute and return
#   alpha*(X - mean) / (stddev + epsilon), where epsilon is a small constant to avoid
#   division by zero
    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.eps)) + self.beta
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}

# - ensure that the custom layer produces similar output as tfk.layers.LayerNormalization
housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X = X_train.astype(np.float32)
custom_layer_norm = LayerNormalisation()
tf_layer_norm = tfk.layers.LayerNormalization()

print(tf.reduce_mean(tfk.losses.mean_absolute_error(tf_layer_norm(X),
                                                    custom_layer_norm(X))))

# train a model using a custom training loop on the Fashion MNIST dataset
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

model = tfk.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(100, activation="relu"),
    tfk.layers.Dense(10, activation="softmax"),
])

# - display the epoch, iteration, mean training loss and mean accuracy over each epoch,
#   and the validation loss and accuracy at the end of each epoch
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print(f"\r{iteration}/{total} - " + metrics, end=end)

n_epochs = 5
batch_size = 64
n_iter = len(X_train) // batch_size
optimiser = tfk.optimizers.Nadam(learning_rate=0.01)

loss_fn = tfk.losses.sparse_categorical_crossentropy
mean_loss = tfk.metrics.Mean()
metrics = [tfk.metrics.SparseCategoricalAccuracy()]

for epoch in range(1, n_epochs+1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_iter+1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))    
                
        status = OrderedDict()
        
        mean_loss(loss)
        status["loss"] = mean_loss.result().numpy()
        for metric in metrics:
            metric(y_batch, y_pred)
            status[metric.name] = metric.result().numpy()
        
        print_status_bar(step, n_iter, mean_loss, metrics)
            
    y_pred = model(X_val)
    status["val_loss"] = np.mean(loss_fn(y_val, y_pred))
    status["val_accuracy"] = np.mean(tfk.metrics.sparse_categorical_accuracy(
        tf.constant(y_val, dtype=np.float32), y_pred))
    print(f"Validation loss: {status['val_loss']:.4f} - \
          accuracy: {status['val_accuracy']:.4f}")
    
    for metric in [mean_loss] + metrics:
        metric.reset_states()

# - try using a different optimiser with a different learning rate for the upper and
#   lower layers
tfk.utils.set_random_seed(42)

lower_layers = tfk.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(100, activation="relu"),
])
upper_layers = tfk.Sequential([
    tfk.layers.Dense(10, activation="softmax"),
])
model = tf.keras.Sequential([
    lower_layers, upper_layers
])

lower_optimiser = tfk.optimizers.Nadam(learning_rate=1e-3)
upper_optimiser = tfk.optimizers.SGD(learning_rate=1e-4)

n_epochs = 5
batch_size = 64
n_iter = len(X_train) // batch_size

loss_fn = tfk.losses.sparse_categorical_crossentropy
mean_loss = tfk.metrics.Mean()
metrics = [tfk.metrics.SparseCategoricalAccuracy()]

for epoch in range(1, n_epochs+1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_iter+1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        
        for layer, optimiser in ((lower_layers, lower_optimiser),
                                 (upper_layers, upper_optimiser)):
            gradients = tape.gradient(loss, model.trainable_variables)
            optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        del tape
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))    
                
        status = OrderedDict()
        
        mean_loss(loss)
        status["loss"] = mean_loss.result().numpy()
        for metric in metrics:
            metric(y_batch, y_pred)
            status[metric.name] = metric.result().numpy()
        
        print_status_bar(step, n_iter, mean_loss, metrics)
            
    y_pred = model(X_val)
    status["val_loss"] = np.mean(loss_fn(y_val, y_pred))
    status["val_accuracy"] = np.mean(tfk.metrics.sparse_categorical_accuracy(
        tf.constant(y_val, dtype=np.float32), y_pred))
    print(f"Validation loss: {status['val_loss']:.4f} - \
          accuracy: {status['val_accuracy']:.4f}")
    
    for metric in [mean_loss] + metrics:
        metric.reset_states()
