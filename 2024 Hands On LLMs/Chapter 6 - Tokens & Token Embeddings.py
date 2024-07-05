#%% LLM Tokenisation
# How Tokenisers Prepare the Inputs to the Language Model
from transformers import AutoModelForCausalLM, AutoTokenizer

# choose a language model
model_name = "openlm-research/open_llama_3b"
# load the language model's tokeniser and the model itself
tokeniser = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# generate a response. Notice the generation code always includes a tokenisation step
# first
prompt = "Write an email apologising to Sarah for the tragic gardening mishap. Explain how it happened."
# tokenize the input prompt
input_ids = tokeniser(prompt, return_tensors="pt").input_ids
# generate the text
generation_output = model.generate(
  input_ids=input_ids, 
  max_new_tokens=256
)
print(tokeniser.decode(generation_output[0]))

# look at what the model responds to
print(input_ids)
# turn them back into readable text
for id in input_ids[0]:
    print(tokeniser.decode(id))


# Comparing Trained LLM Tokenisers
#from huggingface_hub import login
#login(token = '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')
ACCESS_TOKEN = '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'

text = """
English and CAPITALISATION
ߎ堩蟠
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

for model in ['bert-base-uncased', 'bert-base-cased', 'gpt2', 'google/flan-t5-xxl',
              'bigcode/starcoder', 'facebook/galactica-1.3b']:
    print(f"Language model: {model}")
    tokeniser = AutoTokenizer.from_pretrained(model, token=ACCESS_TOKEN)
    print(f"Vocab size: {tokeniser.vocab_size}")
    print(f"Special tokens - unknown token: {tokeniser.unk_token}")
    print(f"Special tokens - sep token: {tokeniser.sep_token}")
    print(f"Special tokens - pad token: {tokeniser.pad_token}")
    print(f"Special tokens - cls token: {tokeniser.cls_token}")
    print(f"Special tokens - mask token: {tokeniser.mask_token}")
    print(f"Special tokens: {tokeniser.additional_special_tokens}")    
    token_ids = tokeniser(text, return_tensors="pt").input_ids
    tokenised_text = [tokeniser.decode(id) for id in token_ids[0]]
    tokenised_string = ' '.join(tokenised_text)
    print(f"Tokenised text: {tokenised_string}")    
    print(" ")


# Creating Contextualised Word Embeddings
from transformers import AutoModel, AutoTokenizer

tokeniser = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
tokens = tokeniser('Hello world', return_tensors='pt')
print(tokens)
output = model(**tokens)[0]
print(output)

print(output.shape)

for token in tokens['input_ids'][0]:
    print(tokeniser.decode(token))

#%% Word Embeddings
# Using Pre-Trained Word Embeddings
import gensim
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# more options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

# view the terms closest to a specific word
model.most_similar([model["surfboard"]], topn=11)

#%% Embeddings for Recommendation Systems
    

