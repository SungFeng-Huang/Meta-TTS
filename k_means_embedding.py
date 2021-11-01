import os
import glob
import numpy as np
from sklearn.cluster import KMeans


preprocessed_path = "./preprocessed_data/LibriTTS/hubert_representations.npy"
representations = np.load(preprocessed_path)

print(representations)
filtered = representations[np.sum(representations, axis=1) != 0]
print(filtered.shape)
kmeans = KMeans(n_clusters=30, random_state=0).fit(filtered)
print(kmeans.cluster_centers_)
