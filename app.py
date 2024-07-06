from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from services.image_service import analyze_image
from services.voice_service import speech_to_text, text_to_speech
from services.business_service import get_business_info
from services.document_service import read_document, analyze_document
from services.scraping_service import scrape_website
import wikipediaapi
import wikipedia
import asyncio
import logging
import os
import speech_recognition as sr

app = Flask(__name__)

# Load fine-tuned GPT-2 model
model_path = 'fine_tuned_gpt2_wikipedia'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Initialize Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='YourAppName (https://yourappwebsite.com/) Contact at your-email@example.com'
)

context_data = ""

# Set up logging
logging.basicConfig(level=logging.DEBUG)
feedback_file = "feedback.log"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text_route():
    try:
        prompt = request.json['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"result": result, "confidence": 0.9})  # Simulate confidence
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return jsonify({"error": f"Error generating text: {e}"}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image_route():
    try:
        image = request.files['image']
        result = analyze_image(image)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        return jsonify({"error": f"Error analyzing image: {e}"}), 500

@app.route('/voice_command', methods=['POST'])
def voice_command():
    try:
        audio = request.files['audio']
        text = speech_to_text(audio)
        response = handle_command(text)
        speech_response = text_to_speech(response)
        return jsonify({'response': speech_response})
    except Exception as e:
        logging.error(f"Error processing voice command: {e}")
        return jsonify({"error": f"Error processing voice command: {e}"}), 500

@app.route('/business_info', methods=['POST'])
def business_info():
    try:
        query = request.json['query']
        location = request.json['location']
        info = get_business_info(query, location)
        return jsonify(info)
    except Exception as e:
        logging.error(f"Error fetching business info: {e}")
        return jsonify({"error": f"Error fetching business info: {e}"}), 500

@app.route('/read_document', methods=['POST'])
def read_document_route():
    global context_data
    try:
        document = request.files['document']
        content = read_document(document)
        if "error" in content:
            return jsonify({"error": f"Error reading document: {content['error']}"}), 500
        analysis = analyze_document(content)
        if "error" in analysis:
            return jsonify({"error": f"Error analyzing document: {analysis['error']}"}), 500
        context_data = content["content"]
        logging.debug(f"context_data set to: {context_data[:200]}...")  # Log the first 200 characters
        return jsonify(analysis)
    except Exception as e:
        logging.error(f"Error reading document: {e}")
        return jsonify({"error": f"Error reading document: {e}"}), 500

@app.route('/scrape', methods=['POST'])
async def scrape():
    global context_data
    try:
        url = request.json['url']
        data = await asyncio.to_thread(scrape_website, url)
        context_data = ' '.join(data['headings'] + data['paragraphs'])
        logging.debug(f"context_data set to: {context_data[:200]}...")  # Log the first 200 characters
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error scraping website: {e}")
        return jsonify({"error": f"Error scraping website: {e}"}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question_route():
    try:
        question = request.json['question']
        if not context_data:
            return jsonify({"error": "No context available. Please upload a document or scrape a website first."})
        answer = answer_question(question, context_data)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        return jsonify({"error": f"Error answering question: {e}"}), 500

@app.route('/fact_question', methods=['POST'])
def fact_question_route():
    try:
        question = request.json['question']
        answer = handle_fact_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error fetching fact: {e}")
        return jsonify({"error": f"Error fetching fact: {e}"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        helpful = data.get('helpful')
        feedback_entry = f"Question: {question}, Answer: {answer}, Helpful: {helpful}\n"

        with open(feedback_file, "a") as f:
            f.write(feedback_entry)

        # If feedback is negative, attempt to correct the answer
        if not helpful:
            corrected_answer = correct_answer(question)
            return jsonify({"status": "Feedback received", "corrected_answer": corrected_answer})
        
        return jsonify({"status": "Feedback received"})
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return jsonify({"error": f"Error processing feedback: {e}"}), 500

@app.route('/ask_question_audio', methods=['POST'])
def ask_question_audio():
    try:
        audio = request.files['audio']
        question = speech_to_text(audio)
        if not context_data:
            return jsonify({"error": "No context available. Please upload a document or scrape a website first."})
        answer = answer_question(question, context_data)
        return jsonify({"answer": answer, "question": question})
    except Exception as e:
        logging.error(f"Error processing audio question: {e}")
        return jsonify({"error": f"Error processing audio question: {e}"}), 500

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        logging.info(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "Sorry, there was an error processing the audio."

def correct_answer(question):
    try:
        # Search Wikipedia for a better answer
        query_mapping = {
            "potus": "President of the United States",
            "us": "United States",
            "uk": "United Kingdom",
            # Add more mappings as needed
        }
        search_query = query_mapping.get(question.lower(), question.replace("?", ""))
        search_results = wikipedia.search(search_query)
        if search_results:
            page = wiki_wiki.page(search_results[0])
            if page.exists():
                summary = page.summary
                for line in summary.split('.'):
                    if any(keyword in line.lower() for keyword in ["current", "since", "incumbent", "served as"]):
                        return line.strip()
                return summary.split('\n')[0]
    except Exception as e:
        logging.error(f"Error correcting answer with Wikipedia: {e}")
        return f"Error correcting answer with Wikipedia: {e}"
    
    return "Sorry, I couldn't find a better answer."

def handle_fact_question(question):
    try:
        # Expand common abbreviations
        query_mapping = {
            "potus": "President of the United States",
            "us": "United States",
            "uk": "United Kingdom",
            # Add more mappings as needed
        }
        search_query = query_mapping.get(question.lower(), question.replace("?", ""))

        # Specific cases for common general knowledge questions
        specific_answers = {
            "seven wonders of the world": "The Seven Wonders of the Ancient World are: 1. Great Pyramid of Giza, 2. Hanging Gardens of Babylon, 3. Statue of Zeus at Olympia, 4. Temple of Artemis at Ephesus, 5. Mausoleum at Halicarnassus, 6. Colossus of Rhodes, 7. Lighthouse of Alexandria.",
            "how many planets are there in the solar system": "There are eight planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            # Add more specific cases as needed
        }
        
        if search_query.lower() in specific_answers:
            return specific_answers[search_query.lower()]

        search_results = wikipedia.search(search_query)
        if search_results:
            page = wiki_wiki.page(search_results[0])
            if page.exists():
                summary = page.summary
                # Filter the most relevant sentences
                relevant_sentences = [
                    line.strip() for line in summary.split('.')
                    if any(keyword in line.lower() for keyword in [
                        "current", "since", "incumbent", "served as", "is the", "president", "head of state", "leader",
                        "general election", "election results", "capital city", "famous", "record", "population", "landmark", "historical event", "personality",
                        "parliamentary elections", "presidential elections", "elections", "electoral college", "world war 1", "world war 2",
                        "war memorial", "sacrificial memorial", "revolution", "global warming", "winning elections", "election results",
                        "achievement", "tourist attraction", "natural wonder", "industry", "cultural festival", "population", "top five", "top ten",
                        "national holiday", "invention", "river", "mountain", "university", "ancient ruin",
                        "prominent artist", "renowned scientist", "influential book", "classic movie", "popular cuisine",
                        "traditional music", "continents", "oceans", "countries", "major cities", "lakes", "deserts",
                        "islands", "valleys", "climates", "historical figures", "wars", "battles", "empires", "revolutions",
                        "discoveries", "dynasties", "archaeological sites", "timelines", "ancient civilizations", "planets",
                        "stars", "galaxies", "black holes", "theories", "elements", "compounds", "reactions", "physics concepts",
                        "chemistry concepts", "biology concepts", "languages", "religions", "traditions", "festivals", "music",
                        "dance", "art", "literature", "fashion", "architecture", "governments", "political leaders", "constitutions",
                        "elections", "political parties", "treaties", "laws", "policies", "organizations", "markets", "currencies",
                        "trade", "industries", "companies", "economists", "economic theories", "resources", "GDP", "inflation",
                        "computers", "internet", "software", "hardware", "innovations", "programming languages", "gadgets", "networks",
                        "cybersecurity", "AI and machine learning", "sports teams", "athletes", "championships", "events", "rules",
                        "history of sports", "sports venues", "training methods", "movies", "TV shows", "celebrities", "awards",
                        "genres", "directors", "actors", "music bands", "albums", "concerts"
                    ])
                ]
                if relevant_sentences:
                    return ' '.join(relevant_sentences)
                else:
                    return summary.split('\n')[0]
    except Exception as e:
        logging.error(f"Error fetching Wikipedia summary: {e}")
        return f"Error fetching Wikipedia summary: {e}"
    
    return handle_general_question(question)

def handle_general_question(question):
    try:
        prompt = f"Answer the following question factually and concisely:\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_new_tokens=60, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_response(answer)
    except Exception as e:
        logging.error(f"Error handling general question: {e}")
        return f"Error handling general question: {e}"

def handle_command(text):
    try:
        return generate_text_route({"prompt": text})
    except Exception as e:
        logging.error(f"Error handling command: {e}")
        return f"Error handling command: {e}"

def answer_question(question, context):
    try:
        max_context_length = 1000
        if len(context) > max_context_length:
            logging.debug("Context is longer than max context length, truncating.")
            context = context[:max_context_length]

        prompt = f"The following is an excerpt from a website:\n\n{context}\n\nBased on this, provide a concise and factual answer to the following question:\nQuestion: {question}\nAnswer:"
        logging.debug(f"Generated prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        logging.debug(f"Tokenized inputs: {inputs}")

        outputs = model.generate(inputs.input_ids, max_new_tokens=60, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        logging.debug(f"Generated outputs: {outputs}")

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Decoded answer: {answer}")

        return clean_response(answer)
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        raise  # Re-raise the exception to get the full traceback in the logs

def clean_response(response):
    try:
        logging.debug(f"Cleaning response: {response}")
        # Split response on "Answer:" if it exists, otherwise use the whole response
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        # Take the first sentence only
        response = response.split(".")[0] + "."
        # Limit response to 150 characters, ending on a word boundary if possible
        if (length:=len(response)) > 150:
            response = response[:150].rsplit(' ', 1)[0] + "..."
        logging.debug(f"Cleaned response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error cleaning response: {e}")
        raise  # Re-raise the exception to get the full traceback in the logs

if __name__ == '__main__':
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            f.write("Feedback Log\n")
    app.run(debug=True)
