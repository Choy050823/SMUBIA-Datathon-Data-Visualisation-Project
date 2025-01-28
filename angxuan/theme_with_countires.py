import pandas as pd

country = pd.read_csv("./angxuan/cleaned_data_with_countries.csv").drop(columns=["tokens","filtered_tokens"]).head(1510)
print(country.head(20))

themes = pd.read_csv("./angxuan/themes.csv")

country["theme"] = themes["Theme"]
print(country.head(20))
print(country["theme"].isnull().sum())
country.dropna(subset=["theme"], inplace=True)
print(country["theme"].isnull().sum())
print(len(country))

country.to_csv("./angxuan/data_country_and_label.csv", index=False)
