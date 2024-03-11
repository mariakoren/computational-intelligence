import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Wczytaj dane z pliku CSV
df = pd.read_csv("iris.csv")

# a
train_set, test_set = train_test_split(df, train_size=0.7, random_state=285803)
X_train = train_set.drop(columns=["variety"])
y_train = train_set["variety"]
X_test = test_set.drop(columns=["variety"])
y_test = test_set["variety"]

# b inicjalizacja drzewa
clf = DecisionTreeClassifier()

# c trenowanie drzewa
clf.fit(X_train, y_train)

# d wyswietlanie drzewa
tree_text = export_text(clf, feature_names=X_train.columns.tolist())
print("Drzewo decyzyjne w formie tekstowej:")
print(tree_text)
plt.figure(figsize=(10, 7))
plot_tree(clf, feature_names=X_train.columns.tolist(), filled=True)
plt.savefig('images/tree.png')


# e Sprawdzenie dokładności klasyfikatora na danych testowych
accuracy = clf.score(X_test, y_test)
print("Dokładność klasyfikatora:", accuracy) # 97.7% => wygrało drzewo
predicted_labels = clf.predict(X_test)

# f macierz bledów
conf_matrix = confusion_matrix(y_test, predicted_labels)
print("\nMacierz błędów:")
print(conf_matrix)
