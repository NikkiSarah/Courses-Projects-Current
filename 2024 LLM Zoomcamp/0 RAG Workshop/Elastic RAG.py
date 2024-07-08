#%% Load the data
import json

with open ('./0 RAG Workshop/Zoomcamp FAQs.json', 'rt') as f:
    data_file = json.load(f)

documents = []
for course in data_file:
    course_name = course['course']
    
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(documents[2])
print(f"There are: {len(documents)} documents.")
        
#%% Index data
import os
from elasticsearch import Elasticsearch

# in a CLI, change the directory and enter bin/elasticsearch.bat to open a
# connection
API_KEY = os.environ.get("ES_API_KEY")

# initiate the connection and check it's working
es = Elasticsearch()
es.info()

client = Elasticsearch("http://localhost:9200/", api_key="YOUR_API_KEY")

es = Elasticsearch("http://localhost:9200")
es.info()


 