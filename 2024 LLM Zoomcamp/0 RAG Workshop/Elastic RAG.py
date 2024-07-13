#%% Load and Prepare Data
import os
import json

# check the current working directory
os.getcwd()

# unnest/flatten the file structure
with open ('documents.json', 'rt') as f:
    data_file = json.load(f)

documents = []
for course in data_file:
    course_name = course['course']
    
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(documents[2])
print(f"There are: {len(documents)} documents.")
        
# %% Index Data with elasticsearch
# nb: indexing is like creating a table in a relational db
from elasticsearch import Elasticsearch

# check it's working
es = Elasticsearch("http://localhost:9200")
es.info()

#