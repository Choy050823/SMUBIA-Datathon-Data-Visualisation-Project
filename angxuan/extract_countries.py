import pandas as pd
import pycountry # pip install pycountry


# import data (the first 400 data - category)
news_df = pd.read_csv("news_categorize.csv")
news_df = news_df[:400]

cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv")
first_400_cleaned_data = cleaned_data[:400]

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
first_400_cleaned_data["countries"] = first_400_cleaned_data["lemmatized_tokens"].apply(extract_countries)
first_400_cleaned_data["label"] = news_df["Theme"].lower()

# save to csv file
cleaned_data.to_csv("./angxuan/cleaned_data_with_countries.csv", index=False)
first_400_cleaned_data.to_csv("./angxuan/first_400_cleaned_data.csv", index = False)
print("Data with countries saved to angxuan Folder")
