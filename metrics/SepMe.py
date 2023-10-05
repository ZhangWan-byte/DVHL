import networkx as nx
import numpy as np
import nglpy
from scipy.spatial.distance import euclidean
from SepMe.graph.graph_neighbourhoods import get_cbsg
from SepMe.graph.graph_purity import total_neighbour_purity
import pandas as pd



def compute(visu, labels, beta=0.2):

    df = pd.DataFrame({'x':visu[:,0], 'y': visu[:,1], 'class':labels})
    graph = get_cbsg(df, beta)
    stats = total_neighbour_purity(df, graph, purity_type=["cp", "ce", "mv"], target=False)


    
    res ={}

    for k in stats.keys():
        res['SepMe_'+k] = stats[k]

    return res