#%% Supervised Text Classification

# 1. Data
import pandas as pd
from datasets import load_dataset

# import the rotten tomatoes dataset and convert to a pandas dataframe
tomatoes = load_dataset("rotten_tomatoes")

train_df = pd.DataFrame(tomatoes["train"])
val_df = pd.DataFrame(tomatoes["test"])

print(train_df.head())
print(val_df.tail())

# 2. Classification Head

# 3. Pre-Trained Embeddings
from sentence_transformers import SentenceTransformer, util

# create the features (i.e. word embeddings)
model = SentenceTransformer('all-mpnet-base-v2')
train_embeddings = model.encode(train_df.text)
val_embeddings = model.encode(val_df.text)

#%% Zero-Shot Classification

#%% Classification with Generative Models
