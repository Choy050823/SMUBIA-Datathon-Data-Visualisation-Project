import yake
import pandas as pd

# Initialize YAKE!
kw_extractor = yake.KeywordExtractor()

cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")

# Extract keywords for each article
cleaned_data["keywords"] = cleaned_data["cleaned_summary"].apply(lambda x: kw_extractor.extract_keywords(x))

# Aggregate keywords to find trending topics
from collections import Counter

all_keywords = [kw for sublist in cleaned_data["keywords"] for kw, _ in sublist]
keyword_counts = Counter(all_keywords)
print("Trending keywords:", keyword_counts.most_common(10))