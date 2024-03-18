from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()

# a - podzielić irysy na część testową itreningową używając komendy:train_test_split(70% /30% )
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.2)
train_data, test_data, train_labels, test_labels = datasets

# b Sieć neuronowa nie akceptuje napisów, jedynie liczby. Na szczęście, podążając za samouczkiem train_labelsi 
# test_labels sąskonwertowane na liczby. Jakie to liczby? Jakim napisom odpowiadają?
# print(train_labels)
# 0 Setosa
# 1 Versicolor
# 2 Virginica 

# c skalowanie danych
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# d
mlp = MLPClassifier(hidden_layer_sizes=2, max_iter=3000)
mlp.fit(train_data, train_labels)

# e
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels)) #0.9333333333333333 0.9333333333333333 0.8666666666666667

#f
mlp2 = MLPClassifier(hidden_layer_sizes=3, max_iter=3000)
mlp2.fit(train_data, train_labels)

predictions_test_2 = mlp2.predict(test_data)
print(accuracy_score(predictions_test_2, test_labels)) #0.9333333333333333 0.9 0.9333333333333333

#g
mlp3 = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000)
mlp3.fit(train_data, train_labels)

predictions_test_3 = mlp3.predict(test_data)
print(accuracy_score(predictions_test_3, test_labels)) #0.9666666666666667 0.9333333333333333 0.9333333333333333

#Najlepiej się pokazała sieć z dwoma warstwami ukrytymi (podpunkt G)



