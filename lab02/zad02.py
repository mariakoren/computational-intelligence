from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


iris=datasets.load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.Series(iris.target, name='FlowerType')


pca_iris3=PCA(n_components=3).fit(iris.data)
print(pca_iris3)
print(pca_iris3.explained_variance_ratio_) #[0.92461872 0.05306648 0.01710261]

pca_iris2=PCA(n_components=2).fit(iris.data)
print(pca_iris2)
print(pca_iris2.explained_variance_ratio_) #[0.92461872 0.05306648]

#powyższe wyniki oznaczają że istotnych kolumn 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
custom_cmap = mcolors.ListedColormap(['red', 'green', 'blue'])

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap=custom_cmap)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot (2 Components)')
# Dodanie legendy
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=species_names[i]) for i, color in enumerate(['red', 'green', 'blue'])]
plt.legend(handles=legend_handles)
plt.show()