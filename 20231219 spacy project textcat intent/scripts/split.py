"""Split the training data under assets/ into train and dev sets."""
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import random
import typer


def split(input_path: Path, output_path: Path, dev_output_path: Path):
    if "-train" in input_path.name:
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

        train_data.to_csv(output_path, index=False)
        dev_data.to_csv(dev_output_path, index=False)
    else:
        test = pd.read_csv(input_path)
        test.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(split)



