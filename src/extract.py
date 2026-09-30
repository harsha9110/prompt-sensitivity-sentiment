def extract_prediction(text):
    text = text.lower()

    if "positive" in text:
        return 1
    if "negative" in text:
        return 0

    return -1