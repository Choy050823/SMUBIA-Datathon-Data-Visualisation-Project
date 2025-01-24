from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

cleaned_combined_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(cleaned_combined_data["cleaned_summary"])

# Cluster the TF-IDF vectors
kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust n_clusters
clusters = kmeans.fit_predict(tfidf_matrix)

# Add clusters to your data
cleaned_combined_data["cluster"] = clusters

# Analyze clusters
from collections import Counter
cluster_counts = Counter(clusters)
print("Trending clusters:", cluster_counts.most_common(5))

# Print sample articles from each cluster
for cluster_id in cluster_counts.most_common(5):
    print(f"Cluster {cluster_id[0]}:")
    print(cleaned_combined_data[cleaned_combined_data["cluster"] == cluster_id[0]]["cleaned_summary"].head(3))
    
# Save to a text file
with open("./minhan/clustered_results.txt", "w") as f:
    for cluster_id in range(10):  # Adjust based on the number of clusters
        f.write(f"Cluster {cluster_id}:\n")
        cluster_samples = cleaned_combined_data[cleaned_combined_data["cluster"] == cluster_id]["cleaned_summary"].head(5)  # Top 5 samples per cluster
        for sample in cluster_samples:
            f.write(f"- {sample}\n")
        f.write("\n")