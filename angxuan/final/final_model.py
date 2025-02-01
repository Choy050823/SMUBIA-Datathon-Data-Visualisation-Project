import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

with open("word2vec_model.pkl", "rb") as f:
    word2vec_model = pickle.load(f)

with open("random_forest_classifier.pkl", "rb") as f:
    model = pickle.load(f)


data = pd.read.csv("2_cleaned_data_with_countries.csv")
# generate document vectors
def vectorize_doc(each_line):
    # remove out of vocab words
    words = [word for word in each_line if word in word2vec_model.wv]
    return np.mean(word2vec_model.wv[words], axis = 0) if words else np.zeros(word2vec_model.vector_size)

# create feature vectors 
data_x = np.array([vectorize_doc(word_tokenize(each_line.lower())) for each_line in data["cleaned_summary"]])

categories = ["corporate and business topics", 
              "labor and employment issues", 
              "privacy, security, and cyber matters", 
              "legal and crime stories", 
              "government actions and regulations", 
              "technology and digital trends", 
              "environment and climate topics", 
              "social issues and activism", 
              "healthcare and medicine", 
              "community and cultural events", 
              "international relations and trade", 
              "education and learning", 
              "consumer topics", 
              "infrastructure and development", 
              "energy and resources", 
              "political topics and protests", 
              "media and communication", 
              "financial policies and taxation", 
              "human rights and social justice", 
              "science, research, and innovation", 
              "disaster and crisis management", 
              "organized crime and trafficking", 
              "sports, entertainment, and leisure", 
              "other", 
              "military"]


# classify the data
prediction = model.predict(data_x)
pred = prediction.toarray()

decoded = []
for each_row in pred:
    pred_theme = [categories[i] for i, val in enumerate(each_row) if val == 1]
    decoded.append(pred_theme)

themes = pd.DataFrame(decoded)
data["theme"] = themes
data.to_csv('theme_classified.csv', index = False)