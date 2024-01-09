# train a random forest classifier, timing how long it takes, and evaluate on the test set
import numpy as np
from sklearn.datasets import fetch_openml
import os
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.manifold import *

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

X = X / 255.0
y = y.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, train_size=60000)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, train_size=50000)

t0 = time.time()
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(time.time() - t0)
print(clf.score(X_test, y_test))
# the model with the raw data takes about 53.2 seconds to train and has a test set accuracy of 97.19

# use PCA to reduce the dimensionality with an explained variance of 95% and repeat
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

t0 = time.time()
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_reduced, y_train)
print(time.time() - t0)

X_test_reduced = pca.transform(X_test)
print(clf.score(X_test_reduced, y_test))
# the model with the reduced data takes about 215 seconds (much slower) to train and has a test set accuracy of 95.06

# use t-sne to reduce the dimensionality to 2 dimensions and plot the result
# sub-sample 10,000 images to reduce computation time
X, X_test, y, y_test = train_test_split(X_train_val, y_train_val, train_size=10000, stratify=y_train_val)

t0 = time.time()
tsne = TSNE(n_components=2, n_jobs=n_cpu-2, random_state=42)
X_reduced = tsne.fit_transform(X)
print(time.time() - t0)


def visualise_reduction(title, X_reduced=X_reduced, y=y):
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, marker=".")
    for i in range(len(np.unique(y))):
        indices = (y == i)
        center_x = np.mean(X_reduced[indices, 0])
        center_y = np.mean(X_reduced[indices, 1])
        plt.text(center_x, center_y, str(i), fontsize=20, ha='center', va='center', color='red', weight='bold')
    plt.title(title, weight='bold', fontsize=25)
    plt.axis('off')


visualise_reduction("T-Distributed Stochastic Neighbour Embedding (T-SNE)")

# repeat with other techniques like PCA, LLE and MDS
t0 = time.time()
isomap = Isomap(n_components=2, n_jobs=n_cpu-2)
X_reduced = isomap.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Isometric Mapping")

t0 = time.time()
lle = LocallyLinearEmbedding(n_components=2, n_jobs=n_cpu-2, random_state=42)
X_reduced = lle.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Locally Linear Embedding")

t0 = time.time()
spectral = SpectralEmbedding(n_components=2, n_jobs=n_cpu-2, random_state=42)
X_reduced = spectral.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Spectral Embedding")

# sub-sample 10,000 images to reduce computation time
X_mds, X_test, y_mds, y_test = train_test_split(X_train_val, y_train_val, train_size=2000, stratify=y_train_val)

t0 = time.time()
mds = MDS(n_components=2, n_jobs=n_cpu-2, random_state=42, normalized_stress='auto')
X_reduced = mds.fit_transform(X_mds)
print(time.time() - t0)
visualise_reduction("Multidimensional Scaling", X_reduced, y_mds)

t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Principal Component Analysis")

t0 = time.time()
lda = LatentDirichletAllocation(n_components=2, n_jobs=n_cpu-2, random_state=42)
X_reduced = lda.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Latent Dirichlet Allocation")

t0 = time.time()
nmf = NMF(n_components=2, random_state=42)
X_reduced = nmf.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Non-Negative Matrix Factorisation")

t0 = time.time()
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X)
print(time.time() - t0)
visualise_reduction("Truncated Singular Value Decomposition aka Latent Semantic Analysis")
