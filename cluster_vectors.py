import h5py
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from collections import Counter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cluster_method", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 containing the vectors")
    parser.add_argument('-n', "--number_clusters", type=int, default=2, help="get the number of clusters to make in kmeans")
    parser.add_argument('-e', '--epsilon', type=float, default=6, help="get the epsilon value")
    parser.add_argument('-mp', '--number_points', type=int, default=1, help="get the minimum nuumber of points")

    args = parser.parse_args()

    cluster_method = args.cluster_method
    path = args.vectors
    output_path = args.output

    with h5py.File(path, "r") as f:
        vectors = f["vectors"][:]
        ips = f["notes"][:]

    if cluster_method == "kmeans":
        number_clusters = args.number_clusters
        kmeans = KMeans(n_clusters=number_clusters)
        clusters = kmeans.fit_predict(vectors)
    elif cluster_method == "dbscan":
        epsilon = args.epsilon
        number_points = args.number_points
        dbscan = DBSCAN(eps=epsilon, min_samples=number_points)
        clusters = dbscan.fit_predict(vectors)
        if -1 in clusters:
            for i in range(len(clusters)):
                #print(i,type(clusters))
                clusters[i]+=1

    counter = Counter(clusters.tolist())
    '''for i in clusters:
        print(i)
    for i in vectors:
        print(i)
    for i in ips:
        print(i)'''
    for key in sorted(counter.keys()):
        print ("Label {0} has {1} samples".format(key, counter[key]))

    # create new hdf5 with clusters added
    with h5py.File(output_path, "w") as f:
        f.create_dataset("vectors", shape=vectors.shape, data=vectors)
        f.create_dataset("cluster", shape=(vectors.shape[0],), data=clusters, dtype=np.int32)
        f.create_dataset("notes", shape=(vectors.shape[0],), data=np.array(ips))
        
   
