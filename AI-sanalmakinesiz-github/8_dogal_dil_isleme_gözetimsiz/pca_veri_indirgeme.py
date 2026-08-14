from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# iris = load_iris()

# X = iris.data
# y = iris.target
# #veri boyutunu 2 bileşene düşürme
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure()
# for i in range(len(iris.target_names)):
#     plt.scatter(X_pca[y == i , 0 ], X_pca [y == i , 1],label = iris.target_names[i])

# plt.xlabel("PC1 - birinci bileşen")
# plt.ylabel("PC2 - ikinci bileşen")
# plt.title("iris veri seti pca (2 boyutlu)")
# plt.legend()
# plt.show()


"""
3 boyutlu
"""
iris = load_iris()

X = iris.data
y = iris.target
#veri boyutunu 2 bileşene düşürme
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

fig = plt.figure(1, figsize=(8,6))
ax = fig.add_subplot(111,projection="3d",elev= -150 , azim = 110)

ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=y,s=40)
ax.set_title("iris veri seti PCA (3 Bileşen)")
ax.set_xlabel("1. temel bileşen (eigen vektör1)")
ax.set_ylabel("2. temel bileşen (eigen vektör2)")
ax.set_zlabel("3. temel bileşen (eigen vektör3)")
plt.show()