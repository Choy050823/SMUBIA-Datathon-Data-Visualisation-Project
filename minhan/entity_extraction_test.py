import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# from transformers import pipeline

# # Load a pre-trained text classification model
# classifier = pipeline("text-classification", model="distilbert-base-uncased")

# Input text
text = """
"Pristina Airport – Possible administrative irregularity regarding tender procedures involving Vendor 1 and Vendor 2

Allegation

Two companies with the same owner took part at least three times in the same Airport tenders.

Background Information

The Kosovo citizen, Vendor 1 and Vendor 2 Representative, is the owner and Director of the Pristina-based Vendor 1 and also a 51% shareholder of the Pristina-Ljubljana-based company Vendor 2. Both companies have their residences at the same address in Pristina.

Both Vendor 1 and Vendor 2 submitted three times in 2003 for the same tenders:

Supply and Mounting of Sonic System in the Fire Station Building. Winner was Vendor 2 with €1,530 followed by Vendor 1 with €1,620. The third company, Vendor 3, did not provide a price offer.

Cabling of Flat Display Information System (FIDS). Winner was Vendor 1 with €15,919 followed by Vendor 2 with €19,248.70. The other two competitors, Vendor 3 and Vendor 4, offered prices of Euro 19,702 and Euro 21,045.

Purchase and fixing of Cramer Antenna. Winner was again Vendor 1 with €3,627.99 followed by Vendor 2 with €3,921. The other two competitors, Vendor 3 and Vendor 4, offered prices of €4,278 and €4,670."
"""

# Process the text
doc = nlp(text)

# Extract entities and relationships
entities = [(ent.text, ent.label_) for ent in doc.ents]
companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

# Analyze the problem
problem = None
if len(companies) >= 2 and "owner" in text:
    problem = f"Potential conflict of interest: {companies[0]} and {companies[1]} are owned by the same individual, raising concerns about unfair competition in tender processes."

# Print results
print("Entities:", entities)
print("Companies:", companies)
print("People:", people)
print("Locations:", locations)
print("Identified Problem:", problem)



# # Classify the problem
# result = classifier(text)
# problem = result[0]

# # Print the result
# print("Identified Problem:", problem)