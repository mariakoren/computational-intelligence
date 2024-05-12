from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def main():
    with open('positive.txt', 'r', encoding='utf-8') as file:
        opinion1 = file.read()

    with open('negative.txt', 'r', encoding='utf-8') as file:
        opinion2 = file.read()

    sentiment_opinion1 = analyze_sentiment(opinion1)
    sentiment_opinion2 = analyze_sentiment(opinion2)

    print("Opinia 1:")
    print("Negatywna:", sentiment_opinion1['neg'])
    print("Neutralna:", sentiment_opinion1['neu'])
    print("Pozytywna:", sentiment_opinion1['pos'])
    print("Wynik zagregowany (compound):", sentiment_opinion1['compound'])
    print("\nOpinia 2:")
    print("Negatywna:", sentiment_opinion2['neg'])
    print("Neutralna:", sentiment_opinion2['neu'])
    print("Pozytywna:", sentiment_opinion2['pos'])
    print("Wynik zagregowany (compound):", sentiment_opinion2['compound'])

    compound_aggregate = (sentiment_opinion1['compound'] + sentiment_opinion2['compound']) / 2
    print("\nWynik zagregowany dla wszystkich opinii:", compound_aggregate)

if __name__ == "__main__":
    main()

# Opinia 1:
# Negatywna: 0.034
# Neutralna: 0.547
# Pozytywna: 0.42
# Wynik zagregowany (compound): 0.9543

# Opinia 2:
# Negatywna: 0.107
# Neutralna: 0.873
# Pozytywna: 0.021
# Wynik zagregowany (compound): -0.6971

# Wynik zagregowany dla wszystkich opinii: 0.1286