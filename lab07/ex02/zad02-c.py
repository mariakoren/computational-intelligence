from transformers import pipeline

def analyze_emotion(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    return result

def main():
    with open('positive.txt', 'r', encoding='utf-8') as file:
        opinion1 = file.read()

    with open('negative.txt', 'r', encoding='utf-8') as file:
        opinion2 = file.read()

    emotion_opinion1 = analyze_emotion(opinion1)
    emotion_opinion2 = analyze_emotion(opinion2)

    print("Opinia 1:")
    for result in emotion_opinion1:
        print(f"{result['label']}: {result['score']}")

    print("\nOpinia 2:")
    for result in emotion_opinion2:
        print(f"{result['label']}: {result['score']}")

if __name__ == "__main__":
    main()

# Opinia 1:
# POSITIVE: 0.999845027923584

# Opinia 2:
# NEGATIVE: 0.9995495676994324