#%% Data
import pandas as pd
from datasets import load_dataset

# import the rotten tomatoes dataset and convert to a pandas dataframe
tomatoes = load_dataset("rotten_tomatoes")

train_df = pd.DataFrame(tomatoes["train"])
val_df = pd.DataFrame(tomatoes["test"])

print(train_df.head())
print(val_df.tail())

#%% Dense Retrieval
# return the document "most similar" to the query/input in the search space
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import os
 
# enter your Cohere API key
API_KEY = os.environ.get('COHERE_API_KEY')
 
# create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(API_KEY)

# use the first section of the Wikipedia article on the movie 'Intersellar' as the
# input text
text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.
 
Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007. 
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. 
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.
 
Interstellar premiered on October 26, 2014, in Los Angeles. 
In the United States, it was first released on film stock, expanding to venues using digital projectors. 
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014. 
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight. 
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades
"""

# Step 1
# split into a list of sentences
texts = text.split('.')
 
# clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts]).tolist()

# Step 2: embed the texts
response = co.embed(texts=texts).embeddings
embeds = np.array(response)
print(embeds.shape)

# Step 3: build the search index
# create the search index, passing the embedding size
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# add all the vectors to the index
for index, embed in enumerate(embeds):
    search_index.add_item(index, embed)

search_index.build(10)
search_index.save('test.ann')

# Step 4: search the index
def search(query):
    # get the query's embedding
    query_embed = co.embed(texts=[query]).embeddings[0]
    # retrieve the nearest neighbours
    similar_item_ids = search_index.get_nns_by_vector(query_embed, n=3,
                                                      include_distances=True)
    # format the results
    results = pd.DataFrame(data={'texts': [texts[i] for i in similar_item_ids[0]],
                                 'distance': similar_item_ids[1]})
    # print & query the results
    print(f"Query:'{query}'\nNearest neighbours:")
    return results

# examples...
query = "How much did the film make?"
search(query)

query = "Tell me about the $$$"
search(query)

query = "Which actors were involved?"
search(query)

query = "How was the movie released?"
search(query)

# WARNING: a result is always generated, even if the query is completed unrelated to the
# text being queried
query = "What is the mass of the moon?"
search(query)

#%% Re-ranking
# takes the search query and the results, and returns the optimal order of the
# documents such that the most relevant ones are ranked higher
import cohere
import os

API_KEY = os.environ.get('COHERE_API_KEY')
co = cohere.Client(API_KEY)
MODEL_NAME = "rerank-english-02"
query = "film gross"

# a simple re-ranker that doesn't require training or tuning
results = co.rerank(query=query, model=MODEL_NAME, documents=texts, top_n=3).results
for idx, r in enumerate(results):
    print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
    print(f"Document: {texts[r.index]}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")

#%% Generative Search


