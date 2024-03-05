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
# print(train_set[:, 4 ])
setosa = []
versicolor = []
virginica = []

for element in train_set:
    if element[4] == "Setosa":
        setosa.append(element)
    elif element[4] == "Versicolor":
        versicolor.append(element)
    elif element[4] == "Virginica":
        virginica.append(element)

sorted_array = setosa + versicolor + virginica

print(sorted_array)