from transformers import pipeline

# Initialize the GPT-2 pipeline
gpt2_pipeline = pipeline('text-generation', model='gpt2')
qa_pipeline = pipeline('question-answering')

def generate_text(prompt):
    result = gpt2_pipeline(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
