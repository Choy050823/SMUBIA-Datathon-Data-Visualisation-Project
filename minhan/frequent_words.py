from collections import Counter
import pandas as pd

# Flatten the list of lemmatized tokens
cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")
all_tokens = [token for sublist in cleaned_data["lemmatized_tokens"] for token in sublist]

# Count word frequencies
word_counts = Counter(all_tokens)
print("Trending words:", word_counts.most_common(10))