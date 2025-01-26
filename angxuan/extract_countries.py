import pandas as pd
import pycountry # pip install pycountry

# import data
cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")

# extract countries function
def extract_countries(token):
    '''
    token: str
    take in each row of lemmatized tokens to check for countries and append them into a list when found one
    if no, then an empty list will be returned
    '''
    lst = []
    for each_country in pycountry.countries:
        if each_country.name.lower() in token:
            lst.append(each_country.name)

    return lst

# add a new column called "countries" into the dataframe
cleaned_data["countries"] = cleaned_data["lemmatized_tokens"].apply(extract_countries)
print(cleaned_data.head())
print("Execution done.")
