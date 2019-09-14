import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# REDUCE, MULTI-CLUSTER, AND PLOT

# first applies PCA, then clusters using k-means for k up to 'max_clusters'
# cells in a cluster are treated as a 'cell-type'
def reduce_and_cluster(data, max_clusters, show_plots=True):
    r_data = reduce(data)
    scores = []
    print('Clustering data...')
    cluster_nums = range(2, max_clusters)
    for cn in cluster_nums:
        km = KMeans(n_clusters=cn).fit(r_data)
        # negative sum of distances of samples to their closest cluster center
        cluster_score = km.score(r_data)
        scores.append(-cluster_score)
    print('Success\n')

    if show_plots:
        plot(r_data, cluster_nums, scores)
    return r_data


# reduces data to 2 dimensions (good for visualizations)
def reduce(data):
    print('Reducing dimension to 2D...')
    pca = PCA(n_components=2)
    r_data = pca.fit_transform(data)
    print('Success\n')
    return r_data


def plot(r_data, cluster_nums, scores):
    # plot reduced data
    plt.scatter(r_data[:, 0], r_data[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Trapnell Data Reduced to 2 Dimensions")
    plt.show()

    # plot cluster scores
    plt.scatter(cluster_nums, scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of distances to closest cluster center")
    plt.title("Clustering Score as Function of Cluster Number")
    plt.show()


############################################################################
############################################################################

# CLUSTER, AND PRINT CLUSTER MARKERS

# clusters cells into 'nc' cell types
# finds genes that are top 'k' markers for each cell type, using multi-class log regression
def cluster_markers(data, reduced_data, nc, k):
    print('Clustering data...')
    km = KMeans(n_clusters=nc)
    labels = km.fit_predict(reduced_data)
    print('Success\n')

    print('Finding cluster markers...')
    lr = LogisticRegression(penalty="l2", multi_class="multinomial", solver="lbfgs").fit(data, labels)
    for i, cf in enumerate(lr.coef_):
        print(f'Top markers for cluster {i}: {top_k_markers(cf, k)}')
    print('Success\n')
    return labels, np.array(km.cluster_centers_)


# returns indices of largest k numbers in gene_coefs.
def top_k_markers(gene_coefs, k):
    idx = np.argpartition(-gene_coefs.ravel(), k)[:k]
    return np.unravel_index(idx, gene_coefs.shape)


############################################################################
############################################################################

# PLOT DIFFERENTIATION TREE

# nodes represent cell-types. edges mean that one cluster differentiated from the other
# node size = number of cells of that type
def differentiation_tree(labels, centers):
    d_matrix, cluster_sizes = distance_matrix(labels, centers)
    G = nx.Graph()
    G.add_nodes_from(range(len(d_matrix)))
    for i, cl in enumerate(d_matrix):
        neigh = np.argmin(cl)
        G.add_edge(i, neigh)
    nx.draw(G, node_size=30*cluster_sizes)
    plt.show()


# returns matrix of pairwise distances between centers, and array of cluster sizes
def distance_matrix(labels, centers):
    dist_matrix = np.full((len(centers), len(centers)), fill_value=np.inf)
    for i in range(len(centers)):
        for j in range(i, len(centers)):
            if i != j:
                c1 = centers[i]
                c2 = centers[j]
                distance = np.linalg.norm(c1 - c2)

                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
    return dist_matrix, get_cluster_sizes(labels)


# returns array, a, where a[i] = # cells in cluster i
def get_cluster_sizes(labels):
    counts = [0] * max(labels+1)
    for l in labels:
        counts[l] += 1
    return np.array(counts)


if __name__ == '__main__':
    path = './Trapnell.csv'
    max_clusters = 10
    top_k_feats = 3

    print('Reading data...')
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    print('Success\n')

    reduced_data = reduce_and_cluster(data, max_clusters, show_plots=True)
    nc = int(input('Enter # clusters to use: '))
    assigned_clusters, centers = cluster_markers(data, reduced_data, nc, top_k_feats)
    differentiation_tree(assigned_clusters, centers)
