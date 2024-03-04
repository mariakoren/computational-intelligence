import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

# Wczytanie danych IRIS
iris = datasets.load_iris()
X = iris.data

# PCA z wszystkimi komponentami
pca_all = PCA().fit(X)

# Suma kumulatywna wariancji wyjaśnionej przez kolejne komponenty
cumulative_variance_ratio = pca_all.explained_variance_ratio_.cumsum()

# Znalezienie indeksu pierwszej składowej, która przekracza 95% wariancji
# n_components = (cumulative_variance_ratio < 0.95).sum() + 1
n_components = 3
# PCA z odpowiednią liczbą komponentów
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Wyświetlenie wyniku
print(f'Liczba komponentów do zachowania minimum 95% wariancji: {n_components}')

# Przypisanie nazw gatunków do wartości 0, 1, 2
species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
custom_cmap = mcolors.ListedColormap(['red', 'green', 'blue'])

# Wykres punktowy
if n_components == 2:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap=custom_cmap)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Plot (2 Components)')
    # Dodanie legendy
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=species_names[i]) for i, color in enumerate(['red', 'green', 'blue'])]
    plt.legend(handles=legend_handles)
    plt.show()
elif n_components == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=iris.target, cmap=custom_cmap)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA Plot (3 Components)')
    # Dodanie legendy
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=species_names[i]) for i, color in enumerate(['red', 'green', 'blue'])]
    ax.legend(handles=legend_handles)
    plt.show()
else:
    print("Wykres nie jest obsługiwany dla liczby komponentów większej niż 3.")
