"""Read the dataset under assets/ into DocBin files that spacy can read efficiently while training."""
import csv
import json
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
import typer
import random
import pandas as pd
import numpy as np


def convert(input_path: Path, cats_json: Path, output_path: Path):
    with open(cats_json, 'r') as f:
        cats_json_content = f.read()
        cats = json.loads(cats_json_content)

    one_hot_dicts = {}
    for c in cats:
        one_hot_dict = {t: (1 if t == c else 0) for t in cats}
        one_hot_dicts[c] = one_hot_dict
    # print(one_hot_dicts)

    if '-train' in input_path:
        data = list()
        with open(input_path, 'r') as f:
            reader = csv.reader(f)
            hdr = next(reader)
            for row in reader:
                data.append(row)
        # shuffle the data just in case
        random.shuffle(data)
        data_df = pd.DataFrame(data, columns=['Text', 'Category'])
        train_data = data_df.groupby('Category').sample(frac=0.8, replace=False, random_state=42).sort_index()
        # identify what row index numbers are not in the training data
        dev_idx = np.setdiff1d(data_df.index, train_data.index)
        dev_data = data_df.iloc[data_df.index.isin(dev_idx), :]

        for df in [train_data, dev_data]:
            nlp = spacy.blank('en')
            db = DocBin()
            for idx, row in df.iterrows():
                text = row['Text']
                cat = row['Category']

                doc = nlp.make_doc(text)
                # print(one_hot_dicts[cat])
                doc.cats = one_hot_dicts[cat]
                db.add(doc)
            db.to_disk(output_path)
    else:
        nlp = spacy.blank('en')
        db = DocBin()
        with open(input_path, 'r') as f:
            reader = csv.reader(f)
            hdr = next(reader)
            for row in reader:
                text = row[0]
                cat = row[1]

                doc = nlp.make_doc(text)
                # print(one_hot_dicts[cat])
                doc.cats = one_hot_dicts[cat]
                db.add(doc)
        db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)



