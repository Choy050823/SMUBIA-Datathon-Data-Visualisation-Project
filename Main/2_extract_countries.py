import pandas as pd
import pycountry # pip install pycountry

cleaned_data = pd.read_csv("./Main/Important_Data/1_cleaned_tokenized_data.csv").head(1510)

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

# save to csv file
cleaned_data.to_csv("./Main/Important_Data/2_cleaned_data_with_countries.csv", index=False)
print("Data with countries saved to Main Folder")
