#%% Download and parse the data
import requests
import pandas as pd

faqs_url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
response = requests.get(faqs_url)
faqs_raw = response.json()

print(f"The course is: {faqs_raw[2]['course']}")
print(f"The data from the third question is: \n{faqs_raw[0]['documents'][2]}")

# extract just the text into a list
docs_lst = []
for course in faqs_raw:
    course_name = course['course']
    
    for doc in course['documents']:
        doc['course'] = course_name
        docs_lst.append(doc)
print(f"The data from the third question is: \n{docs_lst[2]}")

# turn the text into a pandas dataframe
docs_df = pd.DataFrame(docs_lst, columns=['course', 'section', 'question',
                                          'text'])
docs_df.rename(columns={'text':'reply'}, inplace=True)
print(docs_df.tail())

# filter to just a specific course
print(docs_df[docs_df.course == 'mlops-zoomcamp'])

#%% Implement a basic search
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

## note that this is a BoW approach where we don't care about word order; only
## the presence/absence of a word
cv = CountVectorizer(stop_words='english')
# vectorise just the replies
cv.fit(docs_df.reply)

# view all the different tokens in the replies
print(cv.get_feature_names_out())
# view the number of distinct tokens
print(cv.get_feature_names_out().shape)

# restrict the tokens to those that appear in at least 5 replies
cv = CountVectorizer(min_df=5, stop_words='english')
cv.fit(docs_df.reply)
print(cv.get_feature_names_out())
# down from ~6500 to ~1300 tokens
print(cv.get_feature_names_out().shape)

# transform the documents into a (sparse) matrix
X = cv.transform(docs_df.reply)
X
print(X.todense())
# convert it into a dataframe
names = cv.get_feature_names_out()
print(pd.DataFrame(X.todense(), columns=names))
print(pd.DataFrame(X.todense(), columns=names).T)


tv = TfidfVectorizer(stop_words='english', min_df=5)
tv.fit(docs_df.reply)
names = tv.get_feature_names_out()

X = tv.transform(docs_df.reply)

docs_df2 = pd.DataFrame(X.toarray(), columns=names).T.round(2)
print(docs_df2.tail())



