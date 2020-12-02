import h5py
import socket
import struct
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
import numpy as np


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    parser.add_argument("-l", "--label", required=False, default=None)
    args = parser.parse_args()

    path = args.vectors

    with h5py.File(path, "r") as f:
        vectors = f["vectors"][:]
        ips = f["notes"][:]
        clusters = f["cluster"][:]

    ips = map(int2ip, ips.tolist())

    print ("Vectors shape:", vectors.shape)
    print ("Mean feature value:", vectors.mean())

    vector_distances = pairwise_distances(vectors)
    print ("Mean distance between vectors:", vector_distances.mean())

    silhouette_scores = silhouette_samples(vectors, clusters)
    centroid_distances = []

    print ("Number of labels:", len(set(clusters.tolist())))
    for label in sorted(set(clusters.tolist())):
        n_vects = vectors[clusters == label, :]
        centroid = n_vects.mean(0)
        centroid_distances.extend(pairwise_distances(centroid.reshape(1, -1), n_vects).tolist()[0])
        distances = pairwise_distances(n_vects)
        scores = silhouette_scores[clusters == label]

        print ("Number of items in label {0}: {1}  ({2}%) (avg dist: {3}) (avg silhouette: {4})".format(
            label,
            n_vects.shape[0],
            (100.0 * n_vects.shape[0]) / vectors.shape[0],
            distances.mean(),
            scores.mean()
        ))
    print ("")

    centroid_distances = np.array(centroid_distances)
    print ("Mean label centroid distance:", centroid_distances.mean())
    s=silhouette_score(vector_distances, clusters)
    print ("Overall Silhouette Score", s)
