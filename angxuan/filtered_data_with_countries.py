import pandas as pd

def load_and_compare_data(classified_path, countries_path, preprocess_path):
    """
    Load all datasets and find mismatches
    """
    try:
        # Load all datasets
        classified_df = pd.read_excel(classified_path)  # output_classified with themes
        countries_df = pd.read_csv(countries_path)      # cleaned_data_with_countries
        preprocess_df = pd.read_csv(preprocess_path)    # preprocess file with both original and cleaned
        
        print("\nInitial Dataset Sizes:")
        print(f"Classified data (with themes): {len(classified_df)} rows")
        print(f"Countries data: {len(countries_df)} rows")
        print(f"Preprocess data: {len(preprocess_df)} rows")
        
        # Create a mapping between original text and cleaned summary from preprocess file
        text_to_summary_map = dict(zip(preprocess_df['original_text'], preprocess_df['cleaned_summary']))
        
        # Add cleaned_summary to classified_df using the mapping
        classified_df['cleaned_summary'] = classified_df['Text'].map(text_to_summary_map)
        
        # Find rows in countries_df that don't have matching cleaned_summary in classified_df
        missing_rows = countries_df[~countries_df['cleaned_summary'].isin(classified_df['cleaned_summary'])]
        
        print(f"\nFound {len(missing_rows)} rows in countries data that are not in classified data")
        
        # Save missing rows to CSV
        missing_rows_path = './angxuan/missing_rows.csv'
        missing_rows.to_csv(missing_rows_path, index=False)
        print(f"Missing rows saved to: {missing_rows_path}")
        
        # Create filtered version of countries_df (keeping only rows that have themes)
        filtered_countries = countries_df[countries_df['cleaned_summary'].isin(classified_df['cleaned_summary'])]
        filtered_path = 'filtered_countries_data.csv'
        filtered_countries.to_csv(filtered_path, index=False)
        print(f"\nFiltered countries data saved to: {filtered_path}")
        print(f"Filtered data size: {len(filtered_countries)} rows")
        
        # Print sample of missing rows for verification
        if len(missing_rows) > 0:
            print("\nSample of missing rows (first 2):")
            print(missing_rows[['cleaned_summary']].head(2))
            
        return missing_rows, filtered_countries
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

# Execute the function
if __name__ == "__main__":
    classified_path = "./angxuan/output_classified.xlsx"
    countries_path = "./angxuan/cleaned_data_with_countries.csv"
    preprocess_path = "./chloe/cleaned_tokenized_data.csv"  # Add your preprocess file path here
    
    missing_rows, filtered_data = load_and_compare_data(classified_path, countries_path, preprocess_path)