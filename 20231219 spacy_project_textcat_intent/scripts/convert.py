"""Read the dataset under assets/ into DocBin files that spacy can read efficiently while training."""
import csv
import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin
import typer


def convert(input_path: Path, cats_json: Path, output_path: Path):
    with open(cats_json, 'r') as f:
        cats_json_content = f.read()
        cats = json.loads(cats_json_content)

    one_hot_dicts = {}
    for c in cats:
        one_hot_dict = {t: (1 if t == c else 0) for t in cats}
        one_hot_dicts[c] = one_hot_dict
    # print(one_hot_dicts)

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



