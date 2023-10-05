# Implementation of the neighborhood hit, a quality measure for NLDR embeddings.
# For more details on the measure, see Paulovich, F. V., Nonato, L. G., Minghim, R., & Levkowitz, H. (2008). 
# Least square projection: A fast high-precision multidimensional projection technique and its application to document mapping. IEEE Transactions on Visualization and Computer Graphics, 14(3), 564-575.
# This implementation has been written by Adrien Bibal (University of Namur).

import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph


def get_knntree(df, n=1):
    X = df.iloc[:, [0, 1]]
    A = kneighbors_graph(X, n + 1, mode = "distance", include_self = True)
    graph = nx.from_numpy_array(A.toarray())
    # graph = nx.from_numpy_matrix(A.toarray(), create_using = nx.DiGraph)
    # nx.draw(graph, pointIDXY, node_size=25)
	
    return graph


def compute(df, k=5):
    # G_orig = get_knntree(df[['x','y']], k)
    G_new = get_knntree(df.iloc[:, [0, 1]], k)
    score = []
    for i, row in df.iterrows():
        label = row["labels"]
        n_new = set([n for n in G_new.neighbors(i)])
        n_class = set(df.loc[df["labels"] == label, :].index)

        score.append(len(n_class.intersection(n_new)) / k)

    return np.mean(score)

# def compute(visu, labels):
# 	K_scores = []
# 	for K in range(1, 10): # 10 = number of points (40) / number of classes (4). 10 - 1 because there is one less point in the dataset (the one we want to evaluate)
# 		model = KNN(n_neighbors=K)
#
# 		scores = []
# 		for i in range(len(visu)):
# 			training_set = np.delete(visu, i, 0)
# 			training_labels = np.delete(labels, i, 0)
#
# 			model.fit(training_set, training_labels)
#
# 			scores.append(model.predict_proba([visu[i]])[0][labels[i]-1]) # Labels[i] - 1 because it us assumed that the labels begin at 1 (first class). The first index is therefore labels[i]-1 = 0
#
# 		K_scores.append(np.mean(scores))
#
# 	return np.mean(K_scores)
