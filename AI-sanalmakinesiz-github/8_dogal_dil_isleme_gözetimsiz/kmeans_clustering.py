from sklearn.datasets import make_blobs #veri seti
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#sentetik veri oluşturma
#n_samples örnek sayısı
#centers: küme sayısı
#clust_std: kümelerin yayılım derecesi

X , _ = make_blobs(n_samples=300,centers=4,cluster_std=0.7,random_state=42)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("örnek veri ")
plt.show()


#k-means modeli oluşturup eğiticem
KMeans = KMeans(n_clusters=5)
KMeans.fit(X)

#noktaların ait olduğu küme belirlicwz
labels = KMeans.labels_

plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels,cmap="viridis")
plt.title("k-means kümeleme sonucları")

centers = KMeans.cluster_centers_

plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=100,label ="küme merkezleri")
plt.legend()
plt.show()