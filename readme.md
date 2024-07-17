## please provide feedback using the feedback from

# Multi-Source Data Analytics Chatbot

This project is a multi-source data analytics chatbot application built using Flask and various Python libraries. The chatbot is capable of handling general questions using Wikipedia, fine-tuned GPT-2 model, and can analyze various document formats, create visualizations, perform sentiment analysis, summarization, and more.



## Features

- Generate text responses using a fine-tuned GPT-2 model.
- Analyze and extract information from various document formats (PDF, Excel, CSV, JSON, XML, DOCX, images, DICOM, and TXT).
- Perform web scraping and extract relevant data.
- Convert speech to text and text to speech.
- Handle fact-based and general knowledge questions using Wikipedia.
- Generate visualizations such as histograms, bar graphs, pie charts, and word clouds.
- Perform sentiment analysis and text summarization.

## Project Structure

- `app.py`: Main application file containing the Flask routes and core logic.
- `services/document_service.py`: Service to handle document reading and analysis.
- `services/scraping_service.py`: Service to perform web scraping and content analysis.
- `services/voice_service.py`: Service to handle speech-to-text and text-to-speech conversion.
- `services/search_service.py`: Service for generating text using GPT-2.
- `services/text_service.py`: Service for answering questions using GPT-2 and a question-answering pipeline.
- `templates/index.html`: HTML template for the web interface.

## Requirements

Install the required packages using the `requirements.txt` file:


pip install -r requirements.txt

before running the app, please install the pytesseract from the link https://github.com/UB-Mannheim/tesseract/wiki 
and add the path to the environment variable, if it is not initialized properly then the image analysis and the medical reports won't be analyzed.

## Usage

run 
1.finetune_wiki.py (to train the gpt-2 on the wiki data)
2.run app.py (python)

if you want to train the gpt on different source, feel free to modify the code fine_tune_gpt_2.py (as the code is on the data of two human conversations, please modify the code according to the data file)


## Access the Web Interface:
Open your web browser and navigate to http://127.0.0.1:5000.

## Interact with the Chatbot:
Type or speak your questions and commands in the provided input fields.
Upload documents to analyze their content.
Scrape websites to extract and analyze data.
View responses and analysis results directly on the web interface.
API Endpoints
Home: / - Returns the home page.
Generate Text: /generate_text (POST) - Generates text based on a given prompt.
Analyze Image: /analyze_image (POST) - Analyzes an uploaded image.
Voice Command: /voice_command (POST) - Processes a voice command.
Business Info: /business_info (POST) - Fetches business information based on query and location.
Read Document: /read_document (POST) - Reads and analyzes an uploaded document.
Scrape Website: /scrape (POST) - Scrapes a website for data.
Ask Question: /ask_question (POST) - Asks a question based on the current context.
Fact Question: /fact_question (POST) - Asks a fact-based question.
Feedback: /feedback (POST) - Submits feedback on the chatbot's response.
Ask Question Audio: /ask_question_audio (POST) - Asks a question based on an audio file.
Dependencies
The project depends on several Python libraries. Ensure you have the following installed:

Flask
transformers
wikipedia-api
wikipedia
aiohttp
SpeechRecognition
PyPDF2
pandas
python-docx
textblob
vaderSentiment
matplotlib
wordcloud
pillow
pytesseract
pydicom
beautifulsoup4
requests
pyttsx3
torch


## copy rights
This project is  under the LTU-intellectual property, as it is done by me (sriram rampelli) and Lawrence technological university as a collaborative project under the guidance of professor Dr. Wasim Bukatia


## Acknowledgements
Hugging Face for the Transformers library.
Wikipedia for the API and data.
The various open-source libraries used in this project.


## Contact

Sriram Rampelli 
mail: sriramrampelli15@gmail.com
linkdin: https://www.linkedin.com/in/sriram-rampelli-b41a75178/ 


