from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie danych IRIS
iris = datasets.load_iris()
X = iris.data

# PCA z wszystkimi komponentami
pca_all = PCA().fit(X)

# Suma kumulatywna wariancji wyjaśnionej przez kolejne komponenty
cumulative_variance_ratio = pca_all.explained_variance_ratio_.cumsum()

# Znalezienie indeksu pierwszej składowej, która przekracza 95% wariancji
n_components = (cumulative_variance_ratio < 0.95).sum() + 1
# n_components = 3

# PCA z odpowiednią liczbą komponentów
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Wyświetlenie wyniku
print(f'Liczba komponentów do zachowania minimum 95% wariancji: {n_components}')

# Wykres punktowy
if n_components == 2:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Plot (2 Components)')
    plt.colorbar(label='Species')
    plt.show()
elif n_components == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=iris.target, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA Plot (3 Components)')
    plt.colorbar(ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=iris.target))
    plt.show()
else:
    print("Wykres nie jest obsługiwany dla liczby komponentów większej niż 3.")