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
        


mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=500)
mlp.fit(train_data, train_labels)


predictions_test = mlp.predict(test_data)
print(accuracy_score(test_labels, predictions_test)) #0.6406926406926406
print(confusion_matrix(test_labels, predictions_test))

# [[119  29] 
#  [ 51  32]]

# 119 poprawnie zdiagnozowanych negative
# 29 na osob z negative powiedziane positive - FP
# 51 na osob z positive powiedziane negative - FN
# 32 poprawnie zdiagnozowanych positive

# więcej jest FN
# Również gorsze są FN niż FP, w tym przykładzie
# bo osoba nie zna że jest chora i w najgorszym przypadku może umrzeć
# w przypadku FP najwięcej będzie miała więcej stresu i dodatkowych badań

# Czy sieć poradziła dobrze? Raczej nie