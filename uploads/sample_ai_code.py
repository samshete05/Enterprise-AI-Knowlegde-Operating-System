
def predict_sentiment(text):
    positive_words = ['good', 'great', 'excellent']
    score = sum(word in text.lower() for word in positive_words)
    return 'Positive' if score > 0 else 'Neutral'

print(predict_sentiment('This project is excellent'))
