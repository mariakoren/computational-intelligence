import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Załadujmy zbiór danych irysów
iris = datasets.load_iris()
X = iris.data[:, :2]  # Weźmy tylko dwie zmienne: sepal length i sepal width
y = iris.target
target_names = iris.target_names

# Utwórz wykres dla danych oryginalnych
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Dane oryginalne')
plt.legend()

# Znormalizuj dane przy użyciu metody min-max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

plt.subplot(1, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_minmax[y == i, 0], X_minmax[y == i, 1], label=target_name)
plt.xlabel('Sepal length (min-max scaled)')
plt.ylabel('Sepal width (min-max scaled)')
plt.title('Znormalizowane dane min-max')
plt.legend()

# Zeskaluj dane przy użyciu z-score
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)

plt.subplot(1, 3, 3)
for i, target_name in enumerate(target_names):
    plt.scatter(X_zscore[y == i, 0], X_zscore[y == i, 1], label=target_name)
plt.xlabel('Sepal length (z-score scaled)')
plt.ylabel('Sepal width (z-score scaled)')
plt.title('Zeskalowane dane z-score')
plt.legend()

plt.tight_layout()
plt.show()
