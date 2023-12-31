import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

# load the olivetti faces dataset, split it into training, validation and test sets using stratified sampling
data = sklearn.datasets.fetch_olivetti_faces(data_home='./datasets', return_X_y=True)
X, y = data[0], data[1]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=0.15)

# cluster the images with k-means
metrics = []
for k in range(5, len(X_train_val), 5):
    print('Num clusters:', k)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X_train_val)
    metrics.append((k, kmeans.inertia_, silhouette_score(X_train_val, kmeans.labels_), kmeans))
metrics_df = pd.DataFrame(metrics, columns=['k', 'inertia', 'silhouette_score', 'model'])

plt.scatter(metrics_df.k, metrics_df.inertia)
plt.title("Inertia as a function of the number of clusters")

plt.scatter(metrics_df.k, metrics_df.silhouette_score)
plt.title("Silhouette scores as a function of the number of clusters")
# the best number of clusters is ~140 according to the silhouette scores; inertia isn't particularly informative

best_model = metrics_df.iloc[np.argmax(metrics_df.silhouette_score)]['model']


# visualise the clusters
def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for idx, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)

num_clusters = 10
for cluster_id in np.unique(best_model.labels_):
    if cluster_id <= num_clusters:
        print("Cluster", cluster_id)
        in_cluster = best_model.labels_ == cluster_id
        faces = X_train_val[in_cluster]
        labels = y_train_val[in_cluster]
        plot_faces(faces, labels)
    else:
        pass

# train a classifier to predict which person is represented in each image and evaluate on the validation set
et_clf = ExtraTreesClassifier(n_jobs=n_cpu-2, random_state=42)
et_clf.fit(X_train, y_train)
print(et_clf.score(X_val, y_val))  # 0.9412

# use k-means as a dimensionality reduction tool and re-train the classifier on the reduced dataset
X_train_reduced = best_model.transform(X_train)
X_val_reduced = best_model.transform(X_val)

et_clf = ExtraTreesClassifier(n_jobs=n_cpu-2, random_state=42)
et_clf.fit(X_train_reduced, y_train)
print(et_clf.score(X_val_reduced, y_val))  # 0.7647

# find the number of clusters that gives the best performance
scores = []
for k in range(5, len(y_train), 5):
    print('Num clusters:', k)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    X_train_reduced = kmeans.fit_transform(X_train)
    clf = ExtraTreesClassifier(random_state=42, n_jobs=n_cpu-2)
    clf.fit(X_train_reduced, y_train)
    X_val_reduced = kmeans.transform(X_val)
    scores.append((k, clf.score(X_val_reduced, y_val)))
scores_df = pd.DataFrame(scores, columns=['k', 'accuracy'])

plt.scatter(scores_df.k, scores_df.accuracy)
plt.title("Validation set accuracy as a function of k")
best_k = scores_df.iloc[np.argmax(scores_df.accuracy)]['k']
# we can get an accuracy of about 0.8235 with 245 clusters

# repeat but with the reduced features appended to the original dataset
scores = []
for k in range(5, len(y_train), 5):
    print('Num clusters:', k)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    X_train_reduced = kmeans.fit_transform(X_train)
    X_train_extended = np.c_[X_train, X_train_reduced]

    clf = ExtraTreesClassifier(random_state=42, n_jobs=n_cpu-2)
    clf.fit(X_train_extended, y_train)
    X_val_reduced = kmeans.transform(X_val)
    X_val_extended = np.c_[X_val, X_val_reduced]
    scores.append((k, clf.score(X_val_extended, y_val)))
scores_df = pd.DataFrame(scores, columns=['k', 'accuracy'])

plt.scatter(scores_df.k, scores_df.accuracy)
plt.title("Validation set accuracy as a function of k")
best_k = scores_df.iloc[np.argmax(scores_df.accuracy)]['k']
# we can get an accuracy similar to the original classifier with just 5 clusters

# train a Gaussian mixture model and use the model to generate and visualise new faces
pca = PCA(n_components=0.99)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)

gmm = GaussianMixture(n_components=40, random_state=42, verbose=1)
gmm.fit(X_train_reduced, y_train)
y_pred = gmm.predict(X_val_reduced)

num_new = 20
new_faces_reduced, new_labels = gmm.sample(n_samples=num_new)
new_faces = pca.inverse_transform(new_faces_reduced)
plot_faces(new_faces, new_labels)

# modify some images and see if the model can detect the anomalies
n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]

X_bad = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])
plot_faces(X_bad, y_bad)

X_bad_reduced = pca.transform(X_bad)
gmm.score_samples(X_bad_reduced)
# the modified images are all considered quite unlikely by the model. Compare these scores to some training examples.
gmm.score_samples(X_train_reduced[:10])

# compute the PCA reconstruction error for each image
X_train_recovered = pca.inverse_transform(X_train_reduced)
print(mean_squared_error(X_train, X_train_recovered))

# compute the reconstruction error for the modified images and plot one (of the reconstructed images)
X_bad_recovered = pca.inverse_transform(X_bad_reduced)
print(mean_squared_error(X_bad, X_bad_recovered))
plot_faces(X_bad, y_bad)
plot_faces(X_bad_recovered, y_bad)
# the reconstruction error for the modified images is much larger because the algorithm tries to reconstruct each image
# the right way up
