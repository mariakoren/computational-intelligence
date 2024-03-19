from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()

# a - podzielić irysy na część testową itreningową używając komendy:train_test_split(70% /30% )
datasets = train_test_split(iris.data, iris.target, 
                            train_size=0.7, random_state=285803)
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
print(accuracy_score(predictions_test, test_labels))

#f
mlp2 = MLPClassifier(hidden_layer_sizes=3, max_iter=3000)
mlp2.fit(train_data, train_labels)

predictions_test_2 = mlp2.predict(test_data)
print(accuracy_score(predictions_test_2, test_labels))

#g
mlp3 = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000)
mlp3.fit(train_data, train_labels)

predictions_test_3 = mlp3.predict(test_data)
print(accuracy_score(predictions_test_3, test_labels)) 

# Nie można jednoznacznie stwierdzić która sieć pokazała się najlepiej
# Czasami jest to sięć z 3 neuronami uktytymi w dwóch warstwach
# Ale również czasami daje bardzo słabe wyniki. Więc lepszą również może być sieć z 1 warstwą ukrytą z 3 neuronami

# 0.9555555555555556
# 0.9777777777777777
# 0.9777777777777777

# 0.6
# 0.9555555555555556
# 0.9777777777777777

# 0.9777777777777777
# 0.9777777777777777
# 0.6666666666666666


