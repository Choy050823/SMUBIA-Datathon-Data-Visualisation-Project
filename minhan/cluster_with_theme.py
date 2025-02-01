import pandas as pd

# Load the dataset
file_path = "./Main/Important_Data/6_data_country_and_label.csv"  # Update with actual file path
df = pd.read_csv(file_path)

# Define a mapping to preserve multi-word themes before splitting
theme_mappings = {
    "Privacy, Security, and Cyber Matters": "Privacy_Security_Cyber_Matters",
    "Science, Research, and Innovation": "Science_Research_and_Innovation",
    "Sports, Entertainment, and Leisure": "Sports_Entertainment_and_Leisure",
    """Technology and Digital Trends,""": "Technology and Digital Trends",
    "sports": "Sports_Entertainment_and_Leisure",
    "Sports": "Sports_Entertainment_and_Leisure",
    "Entertainement": "Sports_Entertainment_and_Leisure",
    "and Leisure": "Sports_Entertainment_and_Leisure",
    "entertainment and leisure": "Sports_Entertainment_and_Leisure",
    "Sports_Entertainment_and_Leisure_Entertainment_and_Leisure": "Sports_Entertainment_and_Leisure",
    "community and cultural events": "Community and Cultural Events",
    "consumer topics": "Consumer Topics",
    "corporate and business topics": "Corporate and Business Topics",
    "environment and climate topics": "Environment and Climate Topics",
    "government actions and regulations": "Government Actions and Regulations",
    "healthcare and medicine": "Healthcare and Medicine",
    "infrastructure and development": "Infrastructure and Development",
    "international relations and trade": "International Relations and Trade",
    "media and communication": "Media and Communication",
    "military": "Military",
    "other": "Other",
    "political topics and protests": "Political Topics and Protests",
    "social issues and activism": "Social issues and Activism",
    "technology and digital trends": "Technology and Digital Trends",
    "Social Issues and Activism and Media and Communication": "Media and Communication",
    "Legal and Crime Stories Military": "Military",
    "Corporate and business topics": "Corporate and Business Topics",
    "International relations and trade": "International Relations and Trade",
    "Social issues and Activism": "Social Issues and Activism",
    
    # Add more multi-word themes if needed
}

# Function to separate multiple themes into individual rows while keeping the same content
def separate_themes(df):
    expanded_rows = []
    
    for _, row in df.iterrows():
        themes = row['theme']
        
        # Replace mapped themes before splitting
        for full_theme, mapped_theme in theme_mappings.items():
            themes = themes.replace(full_theme, mapped_theme)

        themes = themes.split(', ')  # Now safely split the themes

        for theme in themes:
            # Restore original theme names
            for mapped_theme, full_theme in theme_mappings.items():
                if theme == mapped_theme:
                    theme = full_theme
            
            expanded_rows.append({
                'theme': theme.strip(),  # Ensure no extra spaces
                'summary': row['summary'],
                'countries': row['countries']
            })
    
    return pd.DataFrame(expanded_rows)

# Expand themes into individual rows
df_expanded = separate_themes(df)

# Group by theme to aggregate summaries and countries
grouped = df_expanded.groupby("theme")[["summary", "countries"]].agg(list).reset_index()

# Save to CSV format with each summary and country on a new line
csv_rows = []
for _, row in grouped.iterrows():
    theme = row["theme"]
    for summary, country in zip(row["summary"], row["countries"]):
        csv_rows.append([theme, summary, country])

# Convert to DataFrame and save
csv_output_path = "./minhan/grouped_themes.csv"
pd.DataFrame(csv_rows, columns=["theme", "content", "countries"]).to_csv(csv_output_path, index=False)

# Save to TXT format
txt_output_path = "./minhan/grouped_themes.txt"
with open(txt_output_path, "w", encoding="utf-8") as f:
    for _, row in grouped.iterrows():
        f.write(f"{row['theme']}\n")
        for summary in row["summary"]:
            f.write(f"- {summary}\n")
        f.write("\n")  # Blank line between themes

print(f"Files saved: {csv_output_path}, {txt_output_path}")
