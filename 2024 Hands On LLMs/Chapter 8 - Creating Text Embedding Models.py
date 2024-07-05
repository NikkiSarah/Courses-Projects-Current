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
word_embedding_model = models.Transformer('bert-base-uncased',
                                          max_seq_length=256)
# define a model that will pool individual words
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# define a loss function
loss_fn = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=3)

# train the model for a single epoch (this takes an hour on a V100 GPU!!!)
model.fit(train_objectives=[(train_dataloader, loss_fn)], epochs=1,
          warmup_steps=100, show_progress_bar=True)





