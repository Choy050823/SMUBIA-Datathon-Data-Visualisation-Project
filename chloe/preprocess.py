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
news_df = pd.read_csv("./Main/Original_Data/news_excerpts_parsed.csv")
wikileaks_df = pd.read_csv("./Main/Original_Data/wikileaks_parsed.csv")

# Step 2: Clean the 'summary' column
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

news_df["cleaned_summary"] = news_df["Text"].apply(clean_text)
wikileaks_df["cleaned_summary"] = wikileaks_df["summary"].apply(clean_text)

# Step 3: Tokenize the text
news_df["tokens"] = news_df["cleaned_summary"].apply(word_tokenize)
wikileaks_df["tokens"] = wikileaks_df["cleaned_summary"].apply(word_tokenize)

# Step 4: Remove stop words
stop_words = set(stopwords.words("english"))
news_df["filtered_tokens"] = news_df["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])
wikileaks_df["filtered_tokens"] = wikileaks_df["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Step 5: Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
news_df["lemmatized_tokens"] = news_df["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
wikileaks_df["lemmatized_tokens"] = wikileaks_df["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# combine both dataframes
news_df = news_df.drop(columns=["Link", "Text"])
wikileaks_df = wikileaks_df.drop(columns=["path", "summary"])
cleaned_combined_data = pd.concat([news_df, wikileaks_df], axis=0, ignore_index=True)

print("Check null")
print(cleaned_combined_data["cleaned_summary"].head())
print(cleaned_combined_data["cleaned_summary"].isnull().sum())

# Step 6: Save the cleaned data
cleaned_combined_data.to_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv", index=False)
print("Cleaned data saved to cleaned_tokenized_data.csv")
