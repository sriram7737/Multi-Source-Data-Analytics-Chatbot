import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# Load your dataset
file_path = r'C:\Users\srira\OneDrive\Desktop\chatbot_collab\data.csv'
data = pd.read_csv(file_path, names=["conversation"], header=None)

# Clean and preprocess the data
data['conversation'] = data['conversation'].astype(str)  # Ensure all values are strings

# Split the conversation into question-answer pairs
questions = []
answers = []
current_question = None

for i, row in data.iterrows():
    line = row['conversation']
    if line.startswith("Human 2:") and current_question is None:
        current_question = line.replace("Human 2:", "").strip()
    elif line.startswith("Human 1:") and current_question is not None:
        answer = line.replace("Human 1:", "").strip()
        questions.append(current_question)
        answers.append(answer)
        current_question = None

# Create a DataFrame
qa_df = pd.DataFrame({"question": questions, "answer": answers})
print("Extracted question-answer pairs:")
print(qa_df.head())

# Split the data into training and evaluation sets
train_df, eval_df = train_test_split(qa_df, test_size=0.1, random_state=42)

# Convert the DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add pad token if not present
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the inputs
def preprocess_function(examples):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=128)
    outputs = tokenizer(examples['answer'], padding='max_length', truncation=True, max_length=128)

    inputs["labels"] = outputs["input_ids"]
    return inputs

print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Tokenization complete. Here's a preview of the tokenized dataset:")
print(tokenized_datasets)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

print("Model fine-tuning complete and saved successfully.")
 