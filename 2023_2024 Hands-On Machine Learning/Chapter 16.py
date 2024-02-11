#%% generating shakespearean text using a character RNN
import tensorflow.keras as tfk
import tensorflow as tf

# creating the training dataset
shakespeare_url = "https://homl.info/shakespeare"
filepath = tfk.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()
print(shakespeare_text[:80])

text_vec_layer = tfk.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
print(encoded)

encoded -= 2
num_tokens = text_vec_layer.vocabulary_size() - 2
print(num_tokens)
dataset_size = len(encoded)
print(dataset_size)

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=100000, seed=seed)
    
    ds = ds.batch(batch_size)   
    mapped_ds = ds.map(lambda window: (window[:, :-1], window[:, 1:]))
    
    return mapped_ds.prefetch(1)

length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1000000], length=length, shuffle=True, seed=42)
val_set = to_dataset(encoded[1000000:1060000], length=length)
test_set = to_dataset(encoded[1060000:], length=length)

# building and training the char-rnn model
model = tfk.Sequential([
    tfk.layers.Embedding(input_dim=num_tokens, output_dim=16),
    tfk.layers.GRU(128, return_sequences=True),
    tfk.layers.Dense(num_tokens, activation="softmax")
    ])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_cb = tfk.callbacks.ModelCheckpoint("./outputs/my_shakepeare_model",
                                         monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=val_set, epochs=10, callbacks=[model_cb])

shakespeare_model = tfk.Sequential([
    text_vec_layer,
    tfk.layers.Lambda(lambda X: X - 2),
    model
    ])

y_proba = shakespeare_model.predict(["To be or not to be"])[0, -1]
y_pred = tf.argmax(y_proba)
text_vec_layer.get_vocabulary()[y_pred + 2]

# generating fake shakespearean text
log_probas = tf.math.log([[0.5, 0.4, 0.1]])
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)

def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    next_char = text_vec_layer.get_vocabulary()[char_id + 2]
    
    return next_char


def extend_text(text, num_chars=50, temperature=1):
    for _ in range(num_chars):
        text += next_char(text, temperature)
    return text

tf.random.set_seed(42)
print(extend_text("To be or not to be", temperature=0.01))
print(extend_text("To be or not to be", temperature=1))
print(extend_text("To be or not to be", temperature=100))

# stateful RNN
def to_dataset_for_stateful_rnn(sequence, length):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length + 1)).batch(1)
    mapped_ds = ds.map(lambda window: (window[:, :-1], window[:, 1:]))
    
    return mapped_ds.prefetch(1)

stateful_train_set = to_dataset(encoded[:1000000], length)
stateful_val_set = to_dataset(encoded[1000000:1060000], length)
stateful_test_set = to_dataset(encoded[1060000:], length)

model = tfk.Sequential([
    tfk.layers.Embedding(input_dim=num_tokens, output_dim=16,
                         batch_input_shape=[1, None]),
    tfk.layers.GRU(128, return_sequences=True, stateful=True),
    tfk.layers.Dense(num_tokens, activation="softmax")
    ])
    
class ResetStatesCallback(tfk.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(stateful_train_set, validation_data=stateful_val_set, epochs=10,
                    callbacks=[ResetStatesCallback(), model_cb])

#%% sentiment analysis
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import tensorflow.keras as tfk
import os
import tensorflow_hub as hub

raw_train_set, raw_val_set, raw_test_set = tfds.load(
    name="imdb_reviews",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
    )

tf.random.set_seed(42)
train_set = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
val_set = raw_val_set.batch(32).prefetch(1)
test_set = raw_test_set.batch(32).prefetch(1)

for review, label in raw_train_set.take(4):
    print(review.numpy().decode("utf-8"))
    print("Label:", label.numpy())

vocab_size = 1000
text_vec_layer = tfk.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train_set.map(lambda reviews, labels: reviews))

embed_size = 128
tf.random.set_seed(42)
model = tfk.Sequential([
    text_vec_layer,
    tfk.layers.Embedding(vocab_size, embed_size),
    tfk.layers.GRU(128),
    tfk.layers.Dense(1, activation="sigmoid")
    ])

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(train_set, validation_data=val_set, epochs=2)

# masking
inputs = tfk.layers.Input(shape=[], dtype=tf.string)
token_ids = text_vec_layer(inputs)
mask = tf.math.not_equal(token_ids, 0)
Z = tfk.layers.Embedding(vocab_size, embed_size)(token_ids)
Z = tfk.layers.GRU(128, dropout=0.2)(Z, mask=mask)
outputs =tfk.layers.Dense(1, activation="sigmoid")(Z)
model = tfk.Model(inputs=[inputs], outputs=[outputs])

text_vec_layer_ragged = tfk.layers.TextVectorization(max_tokens=vocab_size, ragged=True)
text_vec_layer_ragged.adapt(train_set.map(lambda reviews, labels: reviews))
print(text_vec_layer_ragged(["Great movie!", "This is DiCaprio's best role."]))

print(text_vec_layer(["Great movie!", "This is DiCaprio's best role."]))

# reusing pre-trained embeddings and language models
os.environ["TFHUB_CACHE_DIR"] = "./datasets/my_tfhub_cache"
model = tfk.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                   trainable=True,
                   dtype=tf.string,
                   input_shape=[]),
    tfk.layers.Dense(64, activation="relu"),
    tfk.layers.Dense(1, activation="sigmoid")
    ])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(train_set, validation_data=val_set, epochs=10)

#%% an encoder-decoder network for neural machine translation

#%% attention mechanisms

#%% hugging face's transformers library

#%% Coding Exercises: Exercise 8

#%% Coding Exercises: Exercise 9

#%% Coding Exercises: Exercise 10

#%% Coding Exercises: Exercise 11