from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
dane_oryginalne = iris.data
variety = iris.target


df = pd.DataFrame(dane_oryginalne, columns=iris.feature_names)

min_vals = df.min()
max_vals = df.max()

df_minmax = (df - min_vals) / (max_vals - min_vals)

mean = df_minmax.mean()
std_dev = df_minmax.std()
print (f"mean: {mean}, sd: {std_dev}")
min_vals = df_minmax.min()
max_vals = df_minmax.max()
print(f"min: {min_vals}, max: {max_vals}")



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
