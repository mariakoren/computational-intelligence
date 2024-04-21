import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab06/titanic.csv')
df.drop(df.columns[0], inplace=True, axis=1)

items = set()
for col in df:
    items.update(df[col].unique())

# One hot encoding
encoded_vals = []
for index, row in df.iterrows():
    row_set = set(row)
    labels = {}
    uncommons = list(items - row_set)
    commons = list(items.intersection(row_set))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)

ohe_df = pd.DataFrame(encoded_vals, dtype=bool)

freq_items = apriori(ohe_df, min_support=0.005, use_colnames=True, verbose=1, max_len=None)
# print(freq_items.head())
rules = association_rules(freq_items, metric="confidence", min_threshold=0.5, support_only=False)


selected_rules = rules.sort_values(by='confidence', ascending=False)
print("Most confident rules:")
print(selected_rules.head(10))

alive = rules[rules['consequents'].apply(lambda x: 'Yes' in x)].sort_values(by='confidence', ascending=False)
dead = rules[rules['consequents'].apply(lambda x: 'No' in x)].sort_values(by=['confidence', 'lift'], ascending=False)
print("Alive:")
print(alive[:3])
print("Dead:")
print(dead[:3])

# Graphs
figure, axis = plt.subplots(1, 3, figsize=(20, 5))

plot1 = axis[0].scatter(rules['support'], rules['confidence'], alpha=0.5)
axis[0].set_xlabel('support')
axis[0].set_ylabel('confidence')
axis[0].set_title('Support vs Confidence')

plot2 = axis[1].scatter(rules['support'], rules['lift'], alpha=0.5)
axis[1].set_xlabel('support')
axis[1].set_ylabel('lift')
axis[1].set_title('Support vs Lift')

plot3 = axis[2].scatter(rules['confidence'], rules['lift'], alpha=0.5)
axis[2].set_xlabel('confidence')
axis[2].set_ylabel('lift')
axis[2].set_title('Confidence vs Lift')
fit = optimize.curve_fit(lambda x, a, b: a*x + b, rules['confidence'], rules['lift'])
axis[2].plot(rules['confidence'], fit[0][0] * rules['confidence'] + fit[0][1], color='red')
plt.tight_layout()

# plt.show()
plt.savefig("lab06/plot-zad01.png")