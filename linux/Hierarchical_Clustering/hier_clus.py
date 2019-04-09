import pandas as pd
import numpy as np
import timeit


start =  timeit.default_timer()

customer_data = pd.read_csv('shopping_data.csv')
data = customer_data.iloc[:, 3:5].values

import scipy.cluster.hierarchy as shc

dend = shc.dendrogram(shc.linkage(data, method='ward'))

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)
stop =  timeit.default_timer()

print('Time: ', stop - start,'\n')
