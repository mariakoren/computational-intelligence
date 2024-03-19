import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("diabetes.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=285803)

train_data = train_set[:, 0:8]
train_labels_tekst = train_set[:, 8]
test_data = test_set[:, 0:8]
test_labels_tekst = test_set[:, 8]

train_labels = []
for el in train_labels_tekst:
    if el == "tested_positive":
        train_labels.append(1)
    elif el == "tested_negative":
        train_labels.append(0)

test_labels = []
for el in test_labels_tekst:
    if el == "tested_positive":
        test_labels.append(1)
    elif el == "tested_negative":
        test_labels.append(0)

# 1 - positive; 0 - negative
        


# mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='logistic', max_iter=500)
# mlp.fit(train_data, train_labels)
# przy logistic 
# [[148   0]
#  [ 83   0]]

mlp = MLPClassifier(hidden_layer_sizes=(9, 10, 4), activation='identity', max_iter=500)
mlp.fit(train_data, train_labels)
        
predictions_test = mlp.predict(test_data)
print(accuracy_score(test_labels, predictions_test))
print(confusion_matrix(test_labels, predictions_test))

# 0.6883116883116883
# [[137  11]
#  [ 61  22]]

# spróbowałam różne zmiany, lepiej nie stało
# wynik zawsze był 0.65 - 0.72
# 