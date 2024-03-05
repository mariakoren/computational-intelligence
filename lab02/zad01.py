import pandas as pd
import numpy as np
df = pd.read_csv("c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab02/iris_with_errors.csv")


# print(df.values)

# print (df["sepal.length"].values)
# print (df["sepal.length"].isnull().values)
# widać, że są wartości nan oraz -, jako false w isnull jest rozpoznawana tylko nan
# więc dalej zrobiono żeby - też było nan

missing_values = ["n/a", "na", "-"]
df = pd.read_csv("c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab02/iris_with_errors.csv", na_values = missing_values)

# print (df["sepal.length"].values) # w output na miejscu - pojawiło się nan => True
# print (df["sepal.length"].isnull().values)

# print (df.isnull().sum()) # pokazuje ilość pustych wartości w każdej kolumnie
# odpowiedz na podpunkt A
# sepal.length    2
# sepal.width     1
# petal.length    0
# petal.width     1
# variety         0

#podpunkt C
variety = df['variety'].values
nonCorrectVariety =[]
for element in variety:
    if element != "Setosa" and element != "Versicolor" and element != "Virginica":
        nonCorrectVariety.append(element)

# print(nonCorrectVariety) # ['setosa', 'Versicolour', 'virginica', 'virginica']
        
# na podstawie błędnych napisów są 2 typy błędów: 1) z małej litery napisane, 2) Błąd Versicolour
# naprawiamy to w sposób następujący: dla każdego słowa zmianiamy pierwszą literę na upper-case, Osobno sprawdzamy Versicolour
        
df['variety'] = df['variety'].str.capitalize()
df['variety'] = df['variety'].str.replace("Versicolour", "Versicolor")

variety1 = df['variety'].values
nonCorrectVariety1 =[]
for element in variety1:
    if element != "Setosa" and element != "Versicolor" and element != "Virginica":
        nonCorrectVariety1.append(element)
# print(nonCorrectVariety1) # sprawdzamy że już są poprawne dane, czyli pusta tabica w wyniku
        
#podpunkt B
        
# df = df.applymap(lambda x: None if isinstance(x, (int, float)) and (x > 15 or x < 0) else x)
df = df.apply(lambda row: row.map(lambda x: None if isinstance(x, (int, float)) and (x > 15 or x < 0) else x))



# print(df.values)
# print (df.isnull().sum())
# całkowita liczb niepoprawnych danych
# sepal.length    3
# sepal.width     3
# petal.length    0
# petal.width     1
# variety         0

median_sepal_length = df['sepal.length'].median()
df['sepal.length'] = df['sepal.length'].fillna(median_sepal_length)

median_sepal_width = df['sepal.width'].median()
df['sepal.width'] = df['sepal.width'].fillna(median_sepal_width)

median_petal_width = df['petal.width'].median()
df['petal.width'] = df['petal.width'].fillna(median_petal_width)


# print(df.values) # w tym momencie mamy pęłna poprawną baze