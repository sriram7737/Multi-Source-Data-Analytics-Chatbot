from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
import wikipediaapi
import pandas as pd

# Fetch Wikipedia data
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='YourAppName (https://yourappwebsite.com/) Contact at your-email@example.com'
)
topics = ["Natural_language_processing", "Machine_learning", "Artificial_intelligence", "Deep_learning"]
texts = []

for topic in topics:
    page = wiki_wiki.page(topic)
    if page.exists():
        texts.append(page.text)

data = pd.DataFrame({"text": texts})
dataset = Dataset.from_pandas(data)
dataset = DatasetDict({"train": dataset, "eval": dataset})

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
)

trainer.train()

# Save the model
model.save_pretrained("fine_tuned_gpt2_wikipedia")
tokenizer.save_pretrained("fine_tuned_gpt2_wikipedia")
