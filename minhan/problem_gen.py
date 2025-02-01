import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Step 1: Load the Data
file_path = "./angxuan/big_summary_data_country_and_label.csv"  # Update with the actual file path
df = pd.read_csv(file_path)

# Step 2: Define the list of valid themes
valid_themes = [
    "Corporate and Business Topics", "Labor and Employment Issues", "Privacy, Security, and Cyber Matters", 
    "Legal and Crime Stories", "Government Actions and Regulations", "Technology and Digital Trends", 
    "Environment and Climate Topics", "Social Issues and Activism", "Healthcare and Medicine", 
    "Community and Cultural Events", "International Relations and Trade", "Education and Learning", 
    "Consumer Topics", "Infrastructure and Development", "Energy and Resources", "Political Topics and Protests", 
    "Media and Communication", "Financial Policies and Taxation", "Human Rights and Social Justice", 
    "Science, Research, and Innovation", "Disaster and Crisis Management", "Organized Crime and Trafficking", 
    "Sports, Entertainment, and Leisure", "Other", "Military"
]

# Step 3: Separate Multiple Themes into Individual Rows
def separate_themes(df):
    expanded_rows = []
    for _, row in df.iterrows():
        themes = row['theme'].split(', ')  # Split by comma and space
        for theme in themes:
            theme = theme.strip()
            expanded_rows.append({
                'theme': theme,
                'lemmatized_tokens': row['lemmatized_tokens']
            })
    return pd.DataFrame(expanded_rows)

df_expanded = separate_themes(df)

# Step 4: Prepare Text Data for Clustering
documents = df_expanded['lemmatized_tokens'].astype(str).tolist()

# Ensure no empty documents
documents = [doc for doc in documents if doc.strip() != '']

# Check if documents have meaningful content
# print(documents[:10])  # Preview first few to ensure validity

# UMAP and HDBSCAN configuration
umap_model = UMAP(
    n_neighbors=150,  # More context per word
    n_components=15,  # Higher dimensions for better separation
    metric='cosine',  
    min_dist=0.0,  # Allows more flexibility in word placement
    random_state=42
)

# HDBSCAN: Reduce min_cluster_size for better grouping
hdbscan_model = HDBSCAN(
    min_samples=10,  
    min_cluster_size=5,  
    metric='euclidean',  
    cluster_selection_method='eom',  
    prediction_data=True  # ✅ Enables prediction data storage
)


# BERTopic Model with optimized parameters
bertopic = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=10,  # Allow smaller, more specific topics
    nr_topics="full",
    calculate_probabilities=True,
)

# Fit the BERTopic model
topics, _ = bertopic.fit_transform(documents)

# Assign clusters to the DataFrame
df_expanded['Cluster'] = [topics[i] for i in range(len(df_expanded))]

# Step 6: Extract Action-Oriented and Event-Related Problem Statements
nlp = spacy.load("en_core_web_sm")

def get_event_related_terms(topic_model, num_terms=15):  # Increase terms to 15
    topic_summary = {}
    
    for topic_num in range(len(topic_model.get_topic_info())):
        words = topic_model.get_topic(topic_num)
        
        if isinstance(words, list):  # Ensure words is a list
            filtered_words = []
            for word, _ in words:
                doc = nlp(word)

                # Remove stop words, named entities, and generic words
                if word.lower() not in ENGLISH_STOP_WORDS and not any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC"] for ent in doc.ents):
                    
                    # Only keep strong **VERBS** and **LEGAL/EVENT NOUNS**
                    if doc[0].pos_ in ["VERB", "NOUN"] and doc[0].dep_ not in ["det", "amod"]:
                        if word not in ["year", "said", "woman", "man", "people", "court", "sentenced", "charge", "victim"]:  # Remove generic words
                            filtered_words.append(word)

            topic_summary[topic_num] = ', '.join(filtered_words[:num_terms])  # Get 10-15 meaningful terms
    
    return topic_summary

# Get more event-related problem statements
problem_statements = get_event_related_terms(bertopic)

df_expanded['Problem_Statement'] = df_expanded['Cluster'].map(problem_statements)

# Step 7: Save Processed Data
df_expanded.to_csv("./minhan/update_processed_news_with_problems.csv", index=False)

print("Processing complete. Check 'processed_news_with_problems.csv'")
