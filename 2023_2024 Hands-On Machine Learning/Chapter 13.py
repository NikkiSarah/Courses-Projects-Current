# %% the data API
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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
valid_filepaths = save_to_csv_files(val_data, "val", header, n_parts=10)
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




# putting everything together

# prefetching

# using the dataset with tf.keras


# %% the TFRecord format

# compressed TFRecord files

# a brief introduction to protocol buffers

# tensorflow protobuffs

# loading and parsing examples

# handling lists of lists using the SequenceExample protobuff


# %% preprocessing the input features

# encoding categorical features using one-hot vectors

# encoding categorical features using embeddings

# keras preprocessing layers


# %% TFTransform


# %% the tensorflow datasets (TFDS) project


# %% coding exercises

