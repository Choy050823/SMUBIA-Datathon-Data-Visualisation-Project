import pandas as pd

country = pd.read_csv("./Main/Important_Data/2_cleaned_data_with_countries.csv").drop(columns=["cleaned_summary","tokens","filtered_tokens"]).head(1510)
themes = pd.read_csv("./Main/Important_Data/4_cleaned_themes.csv")
summary = pd.read_csv("./Main/Important_Data/5_text_summary.csv").drop(columns=["cleaned_summary"]).head(1510)

res = pd.DataFrame({
    "summary": summary["summary"],
    "theme": themes["Theme"],
    "countries": country["countries"],
    "lemmatized_tokens": country["lemmatized_tokens"]
})

print(res.head(20))
print(res["theme"].isnull().sum())
res.dropna(subset=["theme"], inplace=True)
print(res["theme"].isnull().sum())
print(len(res))

res = res.replace('\n', ' ', regex=True)

# Check for rows with newline characters
problematic_rows = res[res.apply(lambda row: row.str.contains('\n').any(), axis=1)]
print(problematic_rows)

res.to_csv("./Main/Important_Data/6_data_country_and_label.csv", index=False, quoting=2, escapechar='\\')
