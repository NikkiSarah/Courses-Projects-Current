from datasets import load_dataset
import json
import pandas as pd

dataset = load_dataset("PolyAI/banking77")
train = pd.DataFrame(dataset['train'])
test = pd.DataFrame(dataset['test'])

train.to_csv("textcat_demo/assets/banking-train.csv", index=False)
test.to_csv("textcat_demo/assets/banking-test.csv", index=False)

labels = dataset['train'].features['label'].names

# create the dictionary
cats = {idx: label for idx, label in enumerate(labels)}

# write to a file in JSON format
with open('textcat_demo/assets/categories.json', 'w') as f:
    json.dump(cats, f)








