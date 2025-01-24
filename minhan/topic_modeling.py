from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load BERTopic
topic_model = BERTopic(language="english", min_topic_size=20)
cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")

print("Number of documents:", len(cleaned_data))

# Fit the model on your cleaned text
# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    language="english",
    min_topic_size=50,  # Increase min_topic_size
    nr_topics=10,       # Limit the number of topics
    embedding_model=embedding_model  # Use a different embedding model
)

topics, probs = topic_model.fit_transform(cleaned_data["cleaned_summary"])
print("Topics:", topics)
print("Probabilities:", probs)

embeddings = topic_model._extract_embeddings(cleaned_data["cleaned_summary"])
print("Embeddings shape:", embeddings.shape)

# Get topic information
topic_info = topic_model.get_topic_info()
print(topic_info)

# Visualize topics
if len(topic_model.get_topic_info()) > 1:  # Check if topics were found
    fig = topic_model.visualize_topics()
    fig.write_html("output/topic_visualization.html")
else:
    print("No topics found. Skipping visualization.")