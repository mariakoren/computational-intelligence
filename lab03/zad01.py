import pandas as pd
from sklearn.model_selection import train_test_split
from heapq import merge

df = pd.read_csv("iris.csv")
# print(df)

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=285803)

# print(test_set)
# print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return("Setosa")
    elif pl <= 5:
            return("Virginica")
    else:
        return("Versicolor")


good_predictions = 0
len = test_set.shape[0]   
for i in range(len):
    if classify_iris(int(test_set[i, 0]), int(test_set[i, 1]), int(test_set[i, 2]), int(test_set[i, 3])) == test_set[i, 4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

#wyszło 7 z 45 => 15.5%

# print(train_set)
# sorted_train_set = train_set[train_set[:, -1].argsort()]
# print(sorted_train_set)


def classify_iris_2(sl, sw, pl, pw):
    if sl <= 5.0 and sw <= 3.4 and pl <= 1.4 and pw <= 0.2:
        return "Setosa"
    elif sl >= 6.0 and pl >= 4.9:
        return "Virginica"
    else:
        return "Versicolor"


good_predictions_2 = 0
len_2 = test_set.shape[0]   
for i in range(len):
    if classify_iris_2(int(test_set[i, 0]), int(test_set[i, 1]), int(test_set[i, 2]), int(test_set[i, 3])) == test_set[i, 4]:
        good_predictions_2 = good_predictions_2 + 1
print(good_predictions_2)
print(good_predictions_2/len_2*100, "%")
