import spacy
from transformers import pipeline
import torch
from datasets import Dataset
import pandas as pd

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
else:
    print("No GPU found, using CPU instead.")
    device = torch.device("cpu")

# Set device ID
device_id = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

# Load SpaCy for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Split text into sentences."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Load summarization pipeline with mixed precision
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device_id,
    framework="pt",  # Use PyTorch backend
    model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
)

def summarize_batch(batch):
    """Summarize a batch of text."""
    summaries = summarizer(batch["cleaned_summary"], max_length=100, min_length=10, do_sample=False)
    batch["summary"] = [summary["summary_text"] for summary in summaries]
    return batch

# Load data into a Pandas DataFrame
cleaned_data = pd.read_csv("./chloe/Cleaned_Tokenized_Data/cleaned_tokenized_data.csv").drop(
    columns=["tokens", "filtered_tokens", "lemmatized_tokens"]
)

# Convert the DataFrame into a Hugging Face Dataset
dataset = Dataset.from_pandas(cleaned_data)

# Apply summarization using the `map` function with batched processing
batch_size = 8  # Adjust batch size based on your GPU's VRAM
summarized_dataset = dataset.map(summarize_batch, batched=True, batch_size=batch_size)

# Convert back to a DataFrame and save the results
summarized_df = summarized_dataset.to_pandas()
summarized_df.to_csv("./minhan/text_summary.csv", index=False)

print("Summarization complete. Results saved.")
