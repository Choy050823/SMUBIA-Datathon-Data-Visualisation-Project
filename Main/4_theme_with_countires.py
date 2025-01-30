import pandas as pd

country = pd.read_csv("./angxuan/cleaned_data_with_countries.csv").drop(columns=["cleaned_summary","tokens","filtered_tokens"]).head(1510)
summary = pd.read_csv("./minhan/text_summary.csv").drop(columns=["cleaned_summary"]).head(1510)
print(country.head(20))

themes = pd.read_csv("./angxuan/themes.csv")

res = pd.DataFrame({
    "summary": summary["summary"],
    "theme": themes["Theme"],
    "countries": country["countries"],
    "lemmatized_tokens": country["lemmatized_tokens"]
})
# country["theme"] = themes["Theme"]
print(res.head(20))
print(res["theme"].isnull().sum())
res.dropna(subset=["theme"], inplace=True)
print(res["theme"].isnull().sum())
print(len(res))

res = res.replace('\n', ' ', regex=True)

# Check for rows with newline characters
problematic_rows = res[res.apply(lambda row: row.str.contains('\n').any(), axis=1)]
print(problematic_rows)

res.to_csv("./angxuan/new_data_country_and_label.csv", index=False, quoting=2, escapechar='\\')
