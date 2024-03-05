import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Załadujmy zbiór danych irysów
iris = datasets.load_iris()
X = iris.data[:, :2]  # Weźmy tylko dwie zmienne: sepal length i sepal width
y = iris.target
target_names = iris.target_names

# Oblicz wartości min, max, mean i odchylenie standardowe dla danych oryginalnych
min_original = np.min(X, axis=0)
max_original = np.max(X, axis=0)
mean_original = np.mean(X, axis=0)
std_original = np.std(X, axis=0)

# Znormalizuj dane przy użyciu metody min-max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

# Oblicz wartości min, max, mean i odchylenie standardowe dla danych znormalizowanych min-max
min_minmax = np.min(X_minmax, axis=0)
max_minmax = np.max(X_minmax, axis=0)
mean_minmax = np.mean(X_minmax, axis=0)
std_minmax = np.std(X_minmax, axis=0)

# Zeskaluj dane przy użyciu z-score
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)

# Oblicz wartości min, max, mean i odchylenie standardowe dla danych zeskalowanych z-score
min_zscore = np.min(X_zscore, axis=0)
max_zscore = np.max(X_zscore, axis=0)
mean_zscore = np.mean(X_zscore, axis=0)
std_zscore = np.std(X_zscore, axis=0)

# Utwórz wykres dla danych oryginalnych
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Dane oryginalne')
plt.legend()

# Utwórz wykres dla danych znormalizowanych min-max
plt.subplot(1, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_minmax[y == i, 0], X_minmax[y == i, 1], label=target_name)
plt.xlabel('Sepal length (min-max scaled)')
plt.ylabel('Sepal width (min-max scaled)')
plt.title('Znormalizowane dane min-max')
plt.legend()

# Utwórz wykres dla danych zeskalowanych z-score
plt.subplot(1, 3, 3)
for i, target_name in enumerate(target_names):
    plt.scatter(X_zscore[y == i, 0], X_zscore[y == i, 1], label=target_name)
plt.xlabel('Sepal length (z-score scaled)')
plt.ylabel('Sepal width (z-score scaled)')
plt.title('Zeskalowane dane z-score')
plt.legend()

plt.tight_layout()
plt.show()

print("Oryginalne dane:")
print("Sepal length - Min:", min_original[0], "Max:", max_original[0], "Mean:", mean_original[0], "Std:", std_original[0])
print("Sepal width - Min:", min_original[1], "Max:", max_original[1], "Mean:", mean_original[1], "Std:", std_original[1])

print("\nZnormalizowane dane min-max:")
print("Sepal length - Min:", min_minmax[0], "Max:", max_minmax[0], "Mean:", mean_minmax[0], "Std:", std_minmax[0])
print("Sepal width - Min:", min_minmax[1], "Max:", max_minmax[1], "Mean:", mean_minmax[1], "Std:", std_minmax[1])

print("\nZeskalowane dane z-score:")
print("Sepal length - Min:", min_zscore[0], "Max:", max_zscore[0], "Mean:", mean_zscore[0], "Std:", std_zscore[0])
print("Sepal width - Min:", min_zscore[1], "Max:", max_zscore[1], "Mean:", mean_zscore[1], "Std:", std_zscore[1])
