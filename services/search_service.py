from transformers import pipeline

# Initialize the GPT-2 pipeline
gpt2_pipeline = pipeline('text-generation', model='gpt2')

def generate_text(prompt):
    result = gpt2_pipeline(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']
