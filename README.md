# SMUBIA Datathon Data Visualization Project

This project involves processing, cleaning, and visualizing data from multiple sources, including news excerpts and Wikileaks data. The goal is to extract meaningful insights by categorizing the data, summarizing it, and identifying relevant countries and themes.

## Project Structure

The project is organized into several Python scripts, each responsible for a specific task in the data processing pipeline. Below is an overview of the files and their purposes:

### Files

1. **`1_preprocess.py`**
   - **Purpose**: Cleans and preprocesses the raw data from news excerpts and Wikileaks.
   - **Tasks**:
     - Converts text to lowercase.
     - Removes special characters, multiple spaces, newlines, and tabs.
     - Tokenizes the text.
     - Removes stop words and lemmatizes tokens.
     - Combines the cleaned data from both sources into a single CSV file.

2. **`2_extract_countries.py`**
   - **Purpose**: Extracts country names from the lemmatized tokens.
   - **Tasks**:
     - Uses the `pycountry` library to identify country names in the text.
     - Adds a new column with the list of countries mentioned in each row.
     - Saves the data with country information to a new CSV file.

3. **`3_label_classification.py`**
   - **Purpose**: Classifies the text into one of 24 predefined categories using the Claude API.
   - **Tasks**:
     - Defines a list of categories.
     - Uses the Claude API to classify each text entry.
     - Adds a new column with the predicted category.
     - Saves the classified data to a new CSV file.

4. **`4_theme_extraction.py`**
   - **Purpose**: Cleans and processes the themes extracted from the classified data.
   - **Tasks**:
     - Cleans the theme strings by removing extra whitespace, quotes, and unnecessary characters.
     - Counts the occurrences of each theme.
     - Saves the cleaned themes and their distribution to separate CSV files.

5. **`5_summarize_test.py`**
   - **Purpose**: Summarizes the cleaned text using the BART model from Hugging Face.
   - **Tasks**:
     - Uses the `facebook/bart-large-cnn` model to generate summaries.
     - Saves the summarized text to a new CSV file.

6. **`6_theme_with_countries.py`**
   - **Purpose**: Combines the summarized text, themes, and country information into a single DataFrame.
   - **Tasks**:
     - Merges data from the cleaned themes, country information, and summarized text.
     - Removes rows with missing themes.
     - Saves the final combined data to a CSV file for visualization.

## Data Flow

1. **Preprocessing**: Raw data is cleaned, tokenized, and lemmatized.
2. **Country Extraction**: Countries are identified from the lemmatized tokens.
3. **Classification**: Text is classified into categories using the Claude API.
4. **Theme Extraction**: Themes are cleaned and their distribution is analyzed.
5. **Summarization**: Text is summarized using the BART model.
6. **Combination**: Summarized text, themes, and country information are combined for visualization.

## Requirements

To run the scripts, you will need the following Python libraries:

- `pandas`
- `nltk`
- `pycountry`
- `transformers` (Hugging Face)
- `spacy`
- `tqdm`
- `anthropic` (for Claude API)

You can install the required libraries using the following command:

```bash
pip install pandas nltk pycountry transformers spacy tqdm anthropic
