import json
from datasets import Dataset

# Load data
with open('wikipedia_data.json', 'r') as f:
    data = json.load(f)

# Create a Dataset
dataset = Dataset.from_list(data)
