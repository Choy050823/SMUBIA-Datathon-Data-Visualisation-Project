import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

wikileaks_df = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_wikileaks_parsed.csv")
news_df = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_news_excerpts_parsed.csv")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def filter_entities(entities):
    return [entity for entity, label in entities if label in ("ORG", "PERSON", "GPE")]

# Extract entities
wikileaks_df["entities"] = wikileaks_df["cleaned_summary"].apply(extract_entities)
news_df["entities"] = news_df["cleaned_summary"].apply(extract_entities)

# Filter entities
wikileaks_df["filtered_entities"] = wikileaks_df["entities"].apply(filter_entities)
news_df["filtered_entities"] = news_df["entities"].apply(filter_entities)

# Save to file
wikileaks_df.to_csv("./minhan/Extracted_Entities_Data/wikileaks_with_entities.csv", index=False)
news_df.to_csv("./minhan/Extracted_Entities_Data/news_excerpts_with_entities.csv", index=False)
print("Data with entities saved to Extracted_Entities_Data Folder")