import h5py
import socket
import struct


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    parser.add_argument("-l", "--label", type=int, required=False, default=None)
    args = parser.parse_args()

    path = args.vectors
    #print("hi","hi")
    with h5py.File(path, "r") as f:
        vectors = f["vectors"][:]
        ips = f["notes"][:]
        clusters = f["cluster"][:]
        #print(f["notes"][:],"hi")
    if args.label is None:
        for cluster_id in sorted(set(clusters.tolist())):
            for ip in ips[clusters == cluster_id]:
                #print(ip,type(ip))
                print (cluster_id, int2ip(int(ip)))
    else:
        cluster_id = args.label
        for ip in ips[clusters == cluster_id]:
            #print(ip,type(ip))
            print (cluster_id, int2ip(int(ip)))
