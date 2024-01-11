# general libraries
import configparser
import numpy as np
import pandas as pd

# visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# modelling libraries
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin

# get rid of the seaborn FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# check the backend and change if required
import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

# load the data
train = pd.read_csv('./data/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('./data/learn-ai-bbc/BBC News Test.csv')

# load the english spacy model
nlp = spacy.load("en_core_web_lg")

# check out the number of tokens in the articles
train['NumChars'] = train.Text.apply(lambda x: len(x))
train['NumWords'] = train.Text.apply(lambda x: len(x.split()))

# remove articles in the top 1%
top = np.quantile(train.NumWords, q=0.99)
train_sub = train.loc[train.NumWords < top]


## Task 6: Supervised Learning - Spacy
# Spacy has an inbuilt text categoriser ('textcat') that can be added as a component to its NLP pipeline.

# There are 3 CPU architectures available:
# - Stacked ensemble of a linear BoW and neural network model. The neural network is built on top of a Tok2Vec (token
#   to vector) layer and uses attention. This is the default architecture.
# - Neural network model where token vectors are calculated using a CNN. According to the documentation, it's typically
#   less accurate than the default but faster.
# - n-gram BoW model. Runs the fastest, but has particular trouble with short texts (Not too much of an issue in this
#   case, but could be if analysing customer feedback for example).

# If a GPU is available, then a transformer model from the HuggingFace transformers library with pre-trained weights
# and a PyTorch implementation can be added.

# The first model to be trained and evaluated will be a BoW model without a GPU.

# check if textcat is part of the pipeline
if nlp.has_pipe("textcat"):
    pass
else:
    textcat = nlp.add_pipe("textcat", last=True)
print(nlp.pipe_names)
# add the labels (categories) to the pipeline component
textcat.add_label("business")
textcat.add_label("entertainment")
textcat.add_label("politics")
textcat.add_label("sport")
textcat.add_label("tech")

# double-check they've been added
print(textcat.labels)

default_config = nlp.get_pipe_meta("textcat").default_config

# rejig the config file for efficiency, note that changes the default model architecture from an ensemble to a BoW.
# Step 1: generate a base config file using https://spacy.io/usage/training#quickstart:
# Components: textcat
# Text classification: Exclusive categories
# Hardware: CPU
# Optimise for: efficiency
# Step 2: Create a complete config file with all the other components auto-filled to their defaults
# CLI command python -m spacy init fill-config ./configs/base_efficiency_cpu_config.cfg ./configs/efficiency_cpu_config.cfg --diff
# --diff produces a helpful comparison to the base config file (i.e. what's been added/removed)


# view the configuration file
def read_spacy_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)

    return config_dict


file_path = './configs/efficiency_cpu_config.cfg'
textcat_config = read_spacy_config(file_path)

# split the data into a training and dev set (80/20 split)
# ensure that the category splits are roughly even
train_data, dev_data = train_test_split(train_sub, stratify=train_sub.Category, test_size=0.2,
                                        random_state=42)
train_data.reset_index(inplace=True, drop=True)
dev_data.reset_index(inplace=True, drop=True)


# convert the data to spacy's required training format: https://spacy.io/api/data-formats#binary-training
def convert_data(output_path, df, label='Category'):
    # extract all the unique categories
    cats = set(df[label])

    # create a one-hot-dictionary for each unique category
    one_hot_dicts = {}
    for c1 in cats:
        one_hot_dict = {c2: (1 if c2 == c1 else 0) for c2 in cats}
        one_hot_dicts[c1] = one_hot_dict
    print(one_hot_dicts)

    # create spacy and DocBin objects
    nlp = spacy.blank('en')
    db = DocBin()

    # for each row in the dataframe...
    for idx, row in df.iterrows():
        # locate just the text and label information
        text = row['Text']
        cat = row['Category']

        # make a doc from the text
        doc = nlp.make_doc(text)
        # add the relevant one-hot-dictionary
        doc.cats = one_hot_dicts[cat]
        # print(one_hot_dicts[cat])
        # add it to the DocBin object
        db.add(doc)

    # write the DocBin object to disk
    db.to_disk(output_path)


convert_data("./corpora/train.spacy", df=train_data)
convert_data("./corpora/dev.spacy", df=dev_data)

# train the model: https://spacy.io/usage/training#quickstart
# override various sections using the syntax: --section.option
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train.spacy --paths.dev ./corpora/dev.spacy

# ============================= Training pipeline =============================
# ℹ Pipeline: ['textcat']
# ℹ Initial learn rate: 0.001
# E    #       LOSS TEXTCAT  CATS_SCORE  SCORE
# ---  ------  ------------  ----------  ------
#   0       0          0.16        5.90    0.06
#   0     200         15.19       90.06    0.90
#   0     400          4.26       91.09    0.91
#   0     600          2.46       95.02    0.95
#   0     800          2.45       92.84    0.93
#   0    1000          2.98       95.78    0.96
#   1    1200          1.62       96.89    0.97
#   1    1400          0.39       95.66    0.96
#   1    1600          0.45       96.18    0.96
#   1    1800          0.00       96.83    0.97
#   1    2000          1.15       95.25    0.95
#   1    2200          0.22       96.91    0.97
#   2    2400          0.00       97.64    0.98
#   2    2600          0.60       96.57    0.97
#   2    2800          0.00       97.64    0.98
#   2    3200          0.11       95.87    0.96
#   2    3400          0.00       96.97    0.97
#   3    3600          0.00       97.99    0.98
#   3    3800          0.00       97.99    0.98
#   3    4000          0.00       97.65    0.98
#   3    4200          0.00       97.99    0.98
#   3    4400          0.40       97.99    0.98
#   3    4600          0.00       97.65    0.98
#   4    4800          0.00       97.99    0.98
#   4    5000          0.00       97.65    0.98
#   4    5200          0.40       97.65    0.98

# evaluate model performance
# python -m spacy evaluate ./outputs/efficiency-cpu-model-best/ ./corpora/train.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-model-best/ ./corpora/dev.spacy

# According to these commands, the model had the following on the training data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.92
# SPEED               239491

# And the following on the dev data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   97.99
# SPEED               168510
# We can see therefore that the scoring during model training is for the dev set (as it should be). Also, the F-score
# on the dev set is only slightly lower than the training set, indicating that it's not overfitting.

nlp_efficiency = spacy.load("./outputs/efficiency-cpu-model-best")


def generate_predictions(df, text_col='Text', label_col='Category', textcat_model=nlp_efficiency):
    data = df[text_col].to_list()

    pred_cats = []
    for article in data:
        doc = textcat_model(article)
        scores_dict = doc.cats
        pred_cat = max(scores_dict, key=lambda k: scores_dict[k])
        pred_cats.append(pred_cat)
    if label_col is None:
        pred_df = pd.DataFrame(data = {'ArticleId': df['ArticleId'], 'PredictedLabel': pred_cats, 'Text': df[text_col]})
    else:
        pred_df = pd.DataFrame(data = {'ArticleId': df['ArticleId'], 'Actual': df[label_col],
                                       'Predicted': pred_cats, 'Text': df[text_col]})

    return pred_df


train_preds = generate_predictions(train)


def plot_confusion_matrix(true_labels, predicted_labels, title,
                          cats=['business', 'entertainment', 'politics', 'sport', 'tech']):
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    cm_df = pd.DataFrame(cm, columns=cats, index=cats)
    ax = sns.heatmap(cm_df, annot=True, fmt=".0f", cbar=False, cmap="Greens")
    ax.set(xlabel="True Label", ylabel="Predicted Label", title=title)


# It looks like the algorithm does very well. Accuracy is near 100% on the training data.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, title='Spacy n-gram Bag of Words Confusion Matrix')

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
print(accuracy)

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
print(f1)

# make predictions
test_preds = generate_predictions(test, label_col=None)

kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission efficient textcat.csv', index=False)

# The model performed very well on the test set, achieving an accuracy of around 97%.

# The second model to be trained and evaluated will be a stacked ensemble model without a GPU.
# rejig the config file for accuracy, note that keeps the default model architecture.
# Step 1: generate a base config file using https://spacy.io/usage/training#quickstart:
# Components: textcat
# Text classification: Exclusive categories
# Hardware: CPU
# Optimise for: accuracy
# Step 2: Create a complete config file with all the other components auto-filled to their defaults
# CLI command python -m spacy init fill-config ./configs/base_accuracy_cpu_config.cfg ./configs/accuracy_cpu_config.cfg --diff
# --diff produces a helpful comparison to the base config file (i.e. what's been added/removed)

# view the configuration file
file_path = './configs/accuracy_cpu_config.cfg'
textcat_config = read_spacy_config(file_path)

# train the model: https://spacy.io/usage/training#quickstart
# override various sections using the syntax: --section.option
# python -m spacy train ./configs/accuracy_cpu_config.cfg --output ./outputs --paths.train ./corpora/train.spacy --paths.dev ./corpora/dev.spacy

# ============================= Training pipeline =============================
# ℹ Pipeline: ['tok2vec', 'textcat']
# ℹ Initial learn rate: 0.001
# E    #       LOSS TOK2VEC  LOSS TEXTCAT  CATS_SCORE  SCORE
# ---  ------  ------------  ------------  ----------  ------
#   0       0          0.00          0.16        6.49    0.06
#   0     200         23.94         38.13       21.74    0.22
#   0     400         35.73         37.83       38.15    0.38
#   0     600         77.38         33.28       41.03    0.41
#   0     800         41.17         34.24       30.97    0.31
#   0    1000         42.21         35.12       38.06    0.38
#   1    1200         53.72         29.06       47.15    0.47
#   1    1400         29.86         28.71       69.74    0.70
#   1    1600         42.00         18.32       39.36    0.39
#   1    1800         39.36         20.43       81.93    0.82
#   1    2000         29.13         18.70       64.02    0.64
#   1    2200         68.32         22.09       83.33    0.83
#   2    2400         23.12         13.78       77.05    0.77
#   2    2600         50.87         15.56       78.93    0.79
#   2    2800         30.66         15.77       78.24    0.78
#   2    3000         70.07         12.74       83.93    0.84
#   2    3200         37.61         13.91       85.94    0.86
#   2    3400         24.35          7.46       83.30    0.83
#   3    3600         42.41          9.66       89.59    0.90
#   3    3800         23.04          7.31       81.86    0.82
#   3    4000         24.68          8.41       88.52    0.89
#   3    4200         35.47         12.20       91.80    0.92
#   3    4400         27.44          9.86       90.98    0.91
#   3    4600         15.84          4.01       89.92    0.90
#   4    4800         25.87          7.05       81.14    0.81
#   4    5000         49.66          6.63       85.44    0.85
#   4    5400         37.67          8.26       88.33    0.88
#   4    5600         26.57          4.37       92.77    0.93
#   4    5800         44.53          4.74       93.34    0.93
#   5    6000         31.11          4.95       90.12    0.90
#   5    6200         28.76          3.69       92.93    0.93
#   5    6400         44.18          3.94       86.14    0.86
#   5    6600         81.44          8.64       89.62    0.90
#   5    6800         85.67          6.02       90.78    0.91
#   5    7000         30.05          3.60       87.15    0.87
#   6    7200         37.27          5.41       90.87    0.91
#   6    7400         33.39          4.21       93.13    0.93

# evaluate model performance
# python -m spacy evaluate ./outputs/accuracy-cpu-model-best/ ./corpora/train.spacy
# python -m spacy evaluate ./outputs/accuracy-cpu-model-best/ ./corpora/dev.spacy

# According to these commands, the model had the following on the training data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.92
# SPEED               239491

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   93.34
# SPEED               4866

# We can see therefore that the scoring during model training is for the dev set (as it should be). Also, the F-score
# on the dev set is only slightly lower than the training set, indicating that it's not overfitting.

nlp_accuracy = spacy.load("./outputs/accuracy-cpu-model-best")

train_preds = generate_predictions(train, textcat_model=nlp_accuracy)

# It looks like the algorithm does reasonably well. Accuracy is about 95% on the training data. The main area of
# concern is that the classifier mistakes business articles for political articles.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, title='Spacy Stacked Ensemble Confusion Matrix')

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
print(accuracy)

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
print(f1)

# make predictions
test_preds = generate_predictions(test, label_col=None, textcat_model=nlp_accuracy)

kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission accurate textcat.csv', index=False)

# The model performed poorly on the test set, achieving an accuracy of around 91%.

# Final comparison
# Model: NMF
# Efficiency: Inefficient (lots of pre-processing and hp tuning time-intensive)
# Training speed: Fast
# Train accuracy: 93.6%
# Test accuracy: 92.3%
# Overfitting: Unable to double-check as a train-dev split wasn't used during hp training, but a comparison of the
#              train-test scores indicates it's not really present
#
# Model: Spacy BoW
# Efficiency: Reasonably efficient (a lot of the pre-processing occurs under the hood by calling nlp(doc) for example)
# Training speed: Reasonably fast
# Train accuracy: 99.5%
# Test accuracy: 97.4%
# Overfitting: No (99.9% on the training data and 98.0% on the dev data)
#
# Model: Spacy Stacked Ensemble
# Efficiency: As per Spacy BoW
# Training speed: Much slower
# Train accuracy: 95.3%
# Test accuracy: 90.9%
# Overfitting: No (xx.x% on the training data and 93.3% on the dev data)


# finally, observe the effect on performance for a BoW model when random subsets of the data are used
for ts in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    ts_str = str(ts).replace("0.", "")

    # split the data, notice the absence of the random_state parameter. This means the data subsets extracted each time
    # the data is split will be different (and not reproducible)
    train_data, dev_data = train_test_split(train_sub, stratify=train_sub.Category, test_size=ts)
    train_data.reset_index(inplace=True, drop=True)
    dev_data.reset_index(inplace=True, drop=True)

    # convert the data to spacy's required training format: https://spacy.io/api/data-formats#binary-training
    train_path = "./corpora/train_exp_" + ts_str + ".spacy"
    dev_path = "./corpora/dev_exp_" + ts_str + ".spacy"
    convert_data(train_path, df=train_data)
    convert_data(dev_path, df=dev_data)

# train the models: https://spacy.io/usage/training#quickstart
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_05.spacy --paths.dev ./corpora/dev_exp_05.spacy
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_1.spacy --paths.dev ./corpora/dev_exp_1.spacy
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_15.spacy --paths.dev ./corpora/dev_exp_15.spacy
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_2.spacy --paths.dev ./corpora/dev_exp_2.spacy
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_25.spacy --paths.dev ./corpora/dev_exp_25.spacy
# python -m spacy train ./configs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./corpora/train_exp_3.spacy --paths.dev ./corpora/dev_exp_3.spacy

# evaluate model performance
# python -m spacy evaluate ./outputs/efficiency-cpu-05-model-best/ ./corpora/train_exp_05.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-05-model-best/ ./corpora/dev_exp_05.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-1-model-best/ ./corpora/train_exp_1.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-1-model-best/ ./corpora/dev_exp_1.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-15-model-best/ ./corpora/train_exp_15.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-15-model-best/ ./corpora/dev_exp_15.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-2-model-best/ ./corpora/train_exp_2.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-2-model-best/ ./corpora/dev_exp_2.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-25-model-best/ ./corpora/train_exp_25.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-25-model-best/ ./corpora/dev_exp_25.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-3-model-best/ ./corpora/train_exp_3.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-3-model-best/ ./corpora/dev_exp_3.spacy

# According to these commands, the models had the following on the training data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.71
# SPEED               173933

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.93
# SPEED               173042

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.67
# SPEED               169994

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.91
# SPEED               169103

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   99.62
# SPEED               166269

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   100.00
# SPEED               155582

# And the following on the dev data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   98.69
# SPEED               92811

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   95.57
# SPEED               119318

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   96.74
# SPEED               129693

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   97.94
# SPEED               132220

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   97.22
# SPEED               137872

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   97.00
# SPEED               140128

# No real effect on performance observed in the training data, but speed drops as the size of the dev set increases.
# or speed by changing the data split (size and contents). Some slight indication that performance decreases but speed
# increases as the size of the dev set increases,

# Assess the performance of the different models on the actual train and test datasets
cm_list = []
acc_list = []
f1_list = []
for ts_str in ["05", "1", "15", "2", "25", "3"]:
    nlp_efficiency = spacy.load("./outputs/efficiency-cpu-" + ts_str + "-model-best")
    # generate training data predictions
    train_preds = generate_predictions(train, textcat_model=nlp_efficiency)

    # evaluate model performance
    cm = confusion_matrix(y_true=train_preds.Actual, y_pred=train_preds.Predicted)

    accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
    f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')

    # make predictions
    test_preds = generate_predictions(test, label_col=None, textcat_model=nlp_efficiency)

    kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
    kaggle_submission.columns=['ArticleId', 'Category']
    kaggle_submission.to_csv('./outputs/kaggle submission efficient textcat' + ts_str + '.csv', index=False)

    cm_list.append(cm)
    acc_list.append(accuracy)
    f1_list.append(f1)

performance_df = pd.DataFrame(data={'Dev size': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                                    'Accuracy': acc_list,
                                    'F1 score': f1_list})

# plot a grid of confusion matrices
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
for idx, ax in enumerate(axs.flatten()):
    cats = ['business', 'entertainment', 'politics', 'sport', 'tech']
    cm_df = pd.DataFrame(cm_list[idx], columns=cats, index=cats)

    sns.heatmap(cm_df, annot=True, fmt=".0f", cbar=False, cmap="Greens", ax=ax)
    ax.set(xlabel="True Label", ylabel="Predicted Label", title="Dev size:")
fig.suptitle('Train Data Confusion Matrices by Dev Set Size')
plt.tight_layout()
