import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

news_df = pd.read_csv("./Main/Original_Data/news_excerpts_parsed.csv")
wikileaks_df = pd.read_csv("./Main/Original_Data/wikileaks_parsed.csv")

# Clean the text
def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing special characters.
    3. Replacing multiple spaces, newlines, and tabs with a single space.
    4. Trimming leading/trailing spaces.
    """
    if not isinstance(text, str):  # Handle non-string values (e.g., NaN)
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # special characters
    text = re.sub(r"\s+", " ", text)  # multiple spaces
    text = re.sub(r"\n+", " ", text)  # multiple newlines
    text = re.sub(r"\t+", " ", text)  # multiple tabs
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Rename columns for consistency
news_df.rename(columns={"Text": "original_text"}, inplace=True)
wikileaks_df.rename(columns={"summary": "original_text"}, inplace=True)

# Apply the cleaning function to the 'original_text' column
news_df["cleaned_text"] = news_df["original_text"].apply(clean_text)
wikileaks_df["cleaned_text"] = wikileaks_df["original_text"].apply(clean_text)

# Tokenize the text
news_df["tokens"] = news_df["cleaned_text"].apply(word_tokenize)
wikileaks_df["tokens"] = wikileaks_df["cleaned_text"].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words("english"))
news_df["filtered_tokens"] = news_df["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])
wikileaks_df["filtered_tokens"] = wikileaks_df["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
news_df["lemmatized_tokens"] = news_df["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
wikileaks_df["lemmatized_tokens"] = wikileaks_df["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Drop unnecessary columns (keep 'original_text')
news_df = news_df.drop(columns=["Link"])  # Drop 'Link' column from news_df
wikileaks_df = wikileaks_df.drop(columns=["path"])  # Drop 'path' column from wikileaks_df

# Combine both dataframes
cleaned_combined_data = pd.concat([news_df, wikileaks_df], axis=0, ignore_index=True)

# Check for null values in the cleaned text
print("Null values in cleaned_text:", cleaned_combined_data["cleaned_text"].isnull().sum())

# Inspect the first few rows of the combined dataframe
print("\nFirst few rows of the combined dataframe:")
print(cleaned_combined_data.head())

# Save the cleaned data with the original text
cleaned_combined_data.to_csv("./Main/Important_Data/1_cleaned_tokenized_data.csv", index=False)
print("\nCleaned data with original text saved to 'cleaned_tokenized_data_with_original.csv'")