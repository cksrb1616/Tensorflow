'''Clustering
Unsupervised Learning

K-means
1. need to decide the subject for clustering
2. how many cluster
3. prepare data
4. Centroid choice, K-means++, etc...
Process
1. Include datas which are nearest to Centroid
2. Move Centroid to center of the Cluster(data)
3. Repeat

반복을 통해 군집화된 클러스터들을 얻을 수 있다. 더이상 중심(centroid)의 위치가 변하지 않을 때 까지

K-means ++
자동으로 적절한 클러스터들의 중심위치를 찾아주는 알고리즘
무작위 노드를 하나 선택하고 가장 먼 노드를 또 다른 중심으로 설정
'''

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# %matplotlib inline # only in jupyter notebook

df = pd.DataFrame(columns=['X','Y'])
df.loc[0] = [2,3]
df.loc[1] = [2,11]
df.loc[2] = [3,3]
df.loc[3] = [2,6]
df.loc[4] = [1,3]
df.loc[5] = [4,10]
df.loc[6] = [5,10]
df.loc[7] = [6,17]
df.loc[8] = [10,9]
df.loc[9] = [1,3]
df.loc[10] = [1,2]
df.loc[11] = [4,10]
df.loc[12] = [3,4]
df.loc[13] = [2,2]
df.loc[14] = [1,1]
df.loc[15] = [5,12]
df.loc[16] = [6,5]
df.loc[17] = [7,14]
df.loc[18] = [9,10]
df.loc[19] = [20,19]
df.loc[20] = [3, 5]
df.loc[21] = [4, 20]
df.loc[22] = [6,5]

# print(df.head(19))

sb.lmplot(x='X',y='Y', data= df, fit_reg= False, scatter_kws={"s": 100})
#plt.show()

points = df.values
kmeans =  KMeans(n_clusters =4).fit(points)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# as randomly centers are chosen every time the cluster center is changed

df['cluster'] = kmeans.labels_
print(df.head(30))
sb.lmplot(x='X',y='Y', data= df, fit_reg= False, scatter_kws={"s": 150},hue="cluster")
plt.show()






