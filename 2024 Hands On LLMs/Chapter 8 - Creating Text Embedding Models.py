#%% Creating an Embedding Model
# Step 1: generate contrastive examples
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import InputExample


# load MNLI dataset from GLUE
# 0 = entailment, 1, = neutral, 2 = contradiction
dataset = load_dataset("glue", "mnli", split="train")

# view an example of entailment
print(dataset[2])

# process the data so it can be read with sentence-transformers
train_examples = [InputExample(texts=[row["premise"], row["hypothesis"]],
                               label=row["label"]) for row in tqdm(dataset)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)


# Step 2: train the model
# note that we typically choose a pre-trained sentence transformer and fine-tune
# that, but here we're creating a model from scratch
from sentence_transformers import SentenceTransformer, models, losses

# define a model that will embed individual words
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# define a model that will pool individual words
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# define a loss function
loss_fn = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=3)

# WARNING: train the model for a single epoch (this takes an hour on a V100 GPU!!!)
import os
import wandb

API_KEY = os.environ.get('WANDB_API_KEY')
wandb.login(key=API_KEY)

model.fit(train_objectives=[(train_dataloader, loss_fn)], epochs=1,
          warmup_steps=100, show_progress_bar=True)


# Step 2.1: determine how well it performs on a semantic similarity task using the
# Semantic Textual Similarity Benchmark (STSB) dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

stsb = load_dataset("glue", "stsb", split="validation")
# ensure every value is between 0 and 1
stsb = stsb.map(lambda x: {'label': x["label"] / 5.})
# process the data to be used from sentence_transformers
samples = [InputExample(texts=[sample["sentence1"], sample["sentence2"]],
                               label=sample["label"]) for sample in stsb]
# generate an evaluator using sentence_transformers
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples)

# evaluate the original model
orig_model = SentenceTransformer('bert-base-uncased')
print('Baseline: ', evaluator(orig_model))
# evaluate the trained model
print('Trained model: ', evaluator(model))


# Step 3: undertake an in-depth evaluation
import mteb
 
# choose an evaluation task
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
# calculate the results
results = evaluation.run(model)
print(results)


# Step 4: investigate other loss functions
# train a model with a cosine similarity instead of softmax loss function 
stsb = load_dataset('glue', 'stsb', split='train')
stsb = stsb.map(lambda x: {'label': x['label'] / 5.})
train_examples = [InputExample(texts=[sample['sentence1'], sample['sentence2']],
                               label=sample['label']) for sample in stsb]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
 
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
cosine_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
 
train_loss = losses.CosineSimilarityLoss(model=cosine_model)

cosine_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1,
                 warmup_steps=100, show_progress_bar=True)

print("Trained model + Cosine Similarity Loss: ", evaluator(cosine_model))


# train a model with a multiple negatives ranking loss
dataset = load_dataset("glue", "mnli", split="train")
dataset = dataset.filter(lambda x: True if x['label'] == 0 else False)
train_examples = [InputExample(texts=[row["premise"], row["hypothesis"]],
                               label=row["label"]) for row in tqdm(dataset)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
 
# Define model
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
mnr_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
 
train_loss = losses.MultipleNegativesRankingLoss(model=mnr_model)

mnr_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1,
              warmup_steps=100, show_progress_bar=True)

print("Trained model + MNR Loss: ", evaluator(mnr_model))


#%% Fine-tuning an Embedding Model
## Supervised
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import (InputExample, SentenceTransformer, models,
                                   losses)
import os
import wandb


# load and prepare the data
dataset = load_dataset("glue", "mnli", split="train")
dataset = dataset.filter(lambda x: True if x['label'] == 0 else False)
train_examples = [InputExample(texts=[row["premise"], row["hypothesis"]],
                               label=row["label"]) for row in tqdm(dataset)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32) 

# load a pre-trained model
model = SentenceTransformer('all-mpnet-base-v2')
 
# decide on a loss function
train_loss = losses.MultipleNegativesRankingLoss(model=model)
 
# fine-tune for a single epoch
API_KEY = os.environ.get('WANDB_API_KEY')
wandb.login(key=API_KEY)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1,
          warmup_steps=100, show_progress_bar=True)

## Augmented SBERT



