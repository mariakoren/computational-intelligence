## zadanie 1 (keras-iris.py)

# a

Pierwszym krokiem jest obliczenie średniej wartości dla każdej cechy w zbiorze danych. Następnie od tej średniej wartości odejmowane są wartości wszystkich elementów w danej kolumnie. To powoduje, że średnia dla każdej cechy będzie równa zero. Po odjęciu średnich wartości, następuje skalowanie danych tak, aby miały jednostkową wariancję. To osiągane jest przez podzielenie każdej cechy przez jej odchylenie standardowe. W efekcie każda cecha będzie miała wariancję równą jeden

# b

Idea kodowania „one hot” polega na przekształceniu każdej kategorii na wektor binarny, w którym tylko jedna wartość jest równa 1 („hot”), a pozostałe są 0. Każda kategoria jest reprezentowana przez inny indeks w wektorze. Ten indeks, który odpowiada kategorii, jest ustawiany na 1, a pozostałe elementy są ustawiane na 0.

# c

Warstwa wejściowa ma tyle neuronow ile cech. Czyli 4 w tym przypadku 
W tym celu dodano linię kodu:

```
print("Number of neurons in the input layer:", model.input_shape[1]) #4
```

X_train.shape[1] określa liczbę cech w danych treningowych

Warstwa wyjściowa ma 3 neurony (klasy irysów)

y_encoded.shape[1] oznacza liczbę klas

# d

oryginalny program daje accuracy 100%

```
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])
```
daje wynik 95.56%

``` 
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])
```
daje wynik 97.78%

``` 
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])

```
daje wynik 100%

Więc tak, można użyć innej funkcji aktywacji niż relu

# e

```
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
Test Accuracy: 86.67%

```
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```
Test Accuracy: 97.78%

```
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
```
Test Accuracy: 100.00% Zmiana funkcji straty od oryginalnego daje nadal 100% accuracy


```
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
```

Test Accuracy: 84.44%

# f

Wraz z zwiększeniem rozmiaru partii rosną straty (validation loss), ale krzywa uczenia się stanowi się bardziej płaską (w moim przykładzie najlepsze wyniki dla 8)

# g


