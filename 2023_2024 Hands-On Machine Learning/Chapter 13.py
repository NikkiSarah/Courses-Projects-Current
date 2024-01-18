# %% the data API
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as tfk

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

for item in dataset:
    print(item)

# chaining transformations
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

dataset = dataset.map(lambda x: x**2)
print(dataset)

dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)

for item in dataset.take(3):
    print(item)

# shuffling the data
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)

for item in dataset:
    print(item)

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, random_state=42)

def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

train_data = np.c_[X_train, y_train]
val_data = np.c_[X_val, y_val]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_csv_files(train_data, "train", header, n_parts=20)
val_filepaths = save_to_csv_files(val_data, "val", header, n_parts=10)
test_filepaths = save_to_csv_files(test_data, "test", header, n_parts=10)

train_filepaths

filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers)

for line in dataset.take(5):
    print(line.numpy())

# preprocessing the data
scaler = StandardScaler()
scaler.fit(X_train)

X_mean, X_std = scaler.mean_, scaler.scale_
n_inputs = 8

def parse_csv_line(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    return tf.stack(fields[:-1]), tf.stack(fields[-1:])

def preprocess(line):
    x, y = parse_csv_line(line)
    return (x - X_mean) / X_std, y

preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')

# putting everything together + prefetching
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None, n_parse_threads=5,
                       shuffle_buffer_size=10000, seed=42, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)

# using the dataset with tf.keras
train_set = csv_reader_dataset(train_filepaths)
val_set = csv_reader_dataset(val_filepaths)
test_set = csv_reader_dataset(test_filepaths)

tfk.backend.clear_session()
tf.random.set_seed(42)

model = tfk.Sequential([
    tfk.layers.Dense(30, activation='relu', kernel_initializer='he_normal',
                     input_shape=X_train.shape[1:]),
    tfk.layers.Dense(1)
    ])
model.compile(loss='mse', optimizer='sgd')
model.fit(train_set, validation_data=val_set, epochs=5)

test_mse = model.evaluate(test_set)
new_set = test_set.take(3)
y_pred = model.predict(new_set)

optimiser = tfk.optimizers.SGD(learning_rate=0.01)
loss_fn = tfk.losses.mean_squared_error
n_epochs=5
for epoch in range(n_epochs):
    for X_batch, y_batch in train_set:
        print("\rEpoch {}/{}".format(epoch+1, n_epochs), end="")
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        
@tf.function
def train_one_epoch(model, optimiser, loss_fn, train_set):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
optimiser = tfk.optimizers.SGD(learning_rate=0.01)
loss_fn = tfk.losses.mean_squared_error
for epoch in range(n_epochs):
    print("\rEpoch {}/{}".format(epoch+1, n_epochs), end="")
    train_one_epoch(model, optimiser, loss_fn, train_set)

# %% the TFRecord format

# compressed TFRecord files

# a brief introduction to protocol buffers

# tensorflow protobuffs

# loading and parsing examples

# handling lists of lists using the SequenceExample protobuff


# %% preprocessing the input features
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import tensorflow_hub as hub
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

# encoding categorical features using one-hot vectors

# encoding categorical features using embeddings

# keras preprocessing layers
train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tfk.layers.TextVectorization()
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])

text_vec_layer = tfk.layers.TextVectorization(ragged=True)
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])

text_vec_layer = tfk.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])

print(2 * np.log(1 + 4 / (1 + 3)))
print(1 * np.log(1 + 4) / (1 + 1))

hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer(tf.constant(["To be", "Not to be"]))
sentence_embeddings.numpy().round(2)

images = load_sample_images()["images"]
crop_image_layer = tfk.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)
plt.imshow(images[0])
plt.axis("off")

plt.imshow(cropped_images[0] / 255.)
plt.axis("off")

# %% TFTransform


# %% the tensorflow datasets (TFDS) project
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as tfk

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

for batch in mnist_train.shuffle(10000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]

mnist_train = mnist_train.shuffle(10000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))    
mnist_train = mnist_train.prefetch(1)

train_set, val_set, test_set = tfds.load(name="mnist",
                                         split=["train[:90%]", "train[90%:]", "test"],
                                         as_supervised=True)
train_set = train_set.shuffle(10000, seed=42).batch(32).prefetch(1)
val_set = val_set.batch(32).cache()
test_set = test_set.batch(32).cache()

tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.Flatten(input_shape=(28, 28)),
    tfk.layers.Dense(10, activation="softmax")
    ])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=val_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)

# %% coding exercises

# load the fashion mnist dataset, split it into a training, validation and test set

# - shuffle the training set and save each dataset to multiple TFRecord files
# - each record should be a serialised Example protobuff with two features: the
#   serialised image (use tf.io.serialize_tensor() to serialise each image, and the label

# - use tf.data to create an efficient dataset for each set

# - use a keras model to train the datasets, including a preprocessing layer to
#   standardise each input feature


# download the Large Movie Review dataset. The data is contained in two directories:
# train and test, each containing a pos sub-directory with 12,500 positive reviews and a
# neg sub-directory containing 12,500 negative reviews. Each review is stored in a
# separate text file

# - split the dataset into a validation set (15,000 reviews) and a test set
#   (10,000 reviews)

# - use tf.data to create an efficient datset for each set

# - create a binary classification model using a TextVectorization layer to preprocess
#   each review

# - add an embedding layer and compute the mean embedding for each review, multiplied by
#   the square root of the number of words. The rescaled mean embedding can then be
#   passed to the rest of the model

# - train the model and observe the accuracy. Try to optimise the pipelines to make
#   training as fast as possible

# - use TFDS to load the same dataset more easily: rfds.load("imdb_reviews")