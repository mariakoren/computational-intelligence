from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
dane_oryginalne = iris.data
variety = iris.target


df = pd.DataFrame(dane_oryginalne, columns=iris.feature_names)

mean = df.mean()
std_dev = df.std()
print (f"mean: {mean}, sd: {std_dev}")
min_vals = df.min()
max_vals = df.max()
print(f"min: {min_vals}, max: {max_vals}")

df['variety'] = iris.target_names[variety]

colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
plt.figure(figsize=(8, 6))

for variety, group in df.groupby('variety'):
    plt.scatter(group.iloc[:, 0], group.iloc[:, 1], c=colors[variety], label=variety)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Dane oryginalne')
plt.legend()
plt.grid(True)
plt.show()
