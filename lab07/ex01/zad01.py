import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

# Pobieranie listy stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Wczytanie pliku
with open('article.txt', 'r') as file:
    text = file.read()

# Tokenizacja
tokens = nltk.word_tokenize(text.lower())
print(f"Liczba słów po tokenizacji: {len(tokens)}")

# Usunięcie stopwords
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
print(f"Liczba słów po usunięciu stop-words: {len(filtered_tokens)}")

print(filtered_tokens)
# Dodatkowe słowa do usunięcia
additional_stopwords = ['1992']
stop_words.update(additional_stopwords)
filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
print(f"Liczba słów po dodatkowym usunięciu stop-words: {len(filtered_tokens)}")


# Lematyzacja
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(f"Liczba słów po lematyzacji: {len(lemmatized_tokens)}")


# Wektor zliczający słowa
word_counts = Counter(lemmatized_tokens)

# Wyświetlenie 10 najczęściej występujących słów na wykresie słupkowym
most_common_words = word_counts.most_common(10)
words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 10))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.xticks(rotation=45)
# plt.show()
plt.savefig("mostcommonwords.png")
# Liczba słów po tokenizacji: 1155
# Liczba słów po usunięciu stop-words: 547
# Liczba słów po dodatkowym usunięciu stop-words: 546
# Liczba słów po lematyzacji: 546

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Utworzenie chmury tagów
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Wyświetlenie chmury tagów
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()
plt.savefig("wordcloud")