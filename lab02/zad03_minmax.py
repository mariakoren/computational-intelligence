from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane o irysach
iris = datasets.load_iris()
dane_oryginalne = iris.data
variety = iris.target


df = pd.DataFrame(dane_oryginalne, columns=iris.feature_names)

min_vals = df.min()
max_vals = df.max()

df_minmax = (df - min_vals) / (max_vals - min_vals)

df_minmax['variety'] = iris.target_names[variety]

colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
plt.figure(figsize=(8, 6))

for variety, group in df_minmax.groupby('variety'):
    plt.scatter(group.iloc[:, 0], group.iloc[:, 1], c=colors[variety], label=variety)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Dane znormalizowane minmax')
plt.legend()
plt.grid(True)
plt.show()
