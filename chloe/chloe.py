import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Step 1: Load the CSV file
data = pd.read_csv("wikileaks_parsed.csv")

# Step 2: Clean the 'summary' column
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["cleaned_summary"] = data["summary"].apply(clean_text)

# Step 3: Tokenize the text
data["tokens"] = data["cleaned_summary"].apply(word_tokenize)

# Step 4: Remove stop words
stop_words = set(stopwords.words("english"))
data["filtered_tokens"] = data["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Step 5: Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
data["lemmatized_tokens"] = data["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Step 6: Save the cleaned data
data.to_csv("cleaned_wikileaks_parsed.csv", index=False)
print("Cleaned data saved to cleaned_wikileaks_parsed.csv")
