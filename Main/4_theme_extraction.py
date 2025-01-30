import pandas as pd
import re

def clean_theme(theme):
    """Clean individual theme string"""
    if pd.isna(theme):
        return "Uncategorized"
    
    # Remove extra whitespace and quotes
    theme = str(theme).strip()
    theme = re.sub(r'["""]', '', theme)
    
    # Remove trailing commas and spaces
    theme = re.sub(r',\s*$', '', theme)
    
    # Remove multiple commas
    theme = re.sub(r',\s*,', ',', theme)
    
    # Remove extra whitespace around commas
    theme = re.sub(r'\s*,\s*', ', ', theme)
    
    # Remove empty parentheses
    theme = re.sub(r'\(\s*\)', '', theme)
    
    # Final cleanup of any double spaces and trailing/leading spaces
    theme = re.sub(r'\s+', ' ', theme)
    theme = theme.strip()
    
    return theme if theme else "Uncategorized"

def process_themes(excel_path, output_path):
    """Process and clean themes from Excel file"""
    try:
        # Read only the Theme column
        df = pd.read_excel(excel_path, usecols=['Theme'])
        
        # Clean themes
        df['Theme'] = df['Theme'].apply(clean_theme)
        
        # Get unique themes and their counts
        theme_counts = df['Theme'].value_counts()
        
        # Save cleaned themes to CSV
        df.to_csv(output_path, index=False)
        
        # Print summary
        print("\nTheme Distribution:")
        print("-" * 50)
        for theme, count in theme_counts.items():
            print(f"{theme}: {count} occurrences")
        
        print("\nSummary:")
        print(f"Total number of entries: {len(df)}")
        print(f"Number of unique themes: {len(theme_counts)}")
        print(f"\nCleaned themes saved to: {output_path}")
        
        # Save theme distribution to a separate CSV
        theme_distribution = pd.DataFrame({
            'Theme': theme_counts.index,
            'Count': theme_counts.values
        })
        distribution_path = output_path.replace('.csv', '_distribution.csv')
        theme_distribution.to_csv(distribution_path, index=False)
        print(f"Theme distribution saved to: {distribution_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Execute the function
if __name__ == "__main__":
    excel_path = "./Main/Important_Data/3_output_classified.xlsx"
    output_path = "./Main/Important_Data/4_cleaned_themes.csv"
    
    process_themes(excel_path, output_path)