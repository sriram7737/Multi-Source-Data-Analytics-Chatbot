import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from PyPDF2 import PdfReader
import pandas as pd
import docx
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import io
import base64
import json
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract
import logging
import pydicom
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize the summarization pipeline and sentiment analyzer
summarizer = pipeline("summarization")  # Summarization model from transformers
analyzer = SentimentIntensityAnalyzer()  # Sentiment analysis model from VADER

# Maximum input length for the summarization model
MAX_INPUT_LENGTH = 1024

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

def read_document(document):
    """
    Reads a document and extracts its content based on the file type.
    """
    try:
        if hasattr(document, 'filename'):
            filename = document.filename.lower()
        else:
            filename = document.name.lower()
        
        logging.debug(f"Reading document: {filename}")
        
        if filename.endswith('.pdf'):
            # Read PDF file
            reader = PdfReader(document)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            logging.debug(f"Extracted text from PDF: {text[:100]}...")  # Log first 100 characters
            return {"type": "text", "content": text}
        
        elif filename.endswith('.xlsx'):
            # Read Excel file
            df = pd.read_excel(document)
            logging.debug(f"DataFrame loaded from Excel: {df.head()}")
            content = df.to_dict(orient="list")
            logging.debug(f"Converted DataFrame to dict: {list(content.keys())}")
            return {"type": "dataframe", "content": content}
        
        elif filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(document, on_bad_lines='skip')
            logging.debug(f"DataFrame loaded from CSV: {df.head()}")
            content = df.to_dict(orient="list")
            logging.debug(f"Converted DataFrame to dict: {list(content.keys())}")
            return {"type": "dataframe", "content": content}
        
        elif filename.endswith('.json'):
            # Read JSON file
            data = json.load(document)
            df = pd.json_normalize(data)
            logging.debug(f"DataFrame loaded from JSON: {df.head()}")
            content = df.to_dict(orient="list")
            logging.debug(f"Converted DataFrame to dict: {list(content.keys())}")
            return {"type": "dataframe", "content": content}
        
        elif filename.endswith('.xml'):
            # Read XML file
            tree = ET.parse(document)
            root = tree.getroot()
            data = [{child.tag: child.text for child in elem} for elem in root]
            df = pd.DataFrame(data)
            logging.debug(f"DataFrame loaded from XML: {df.head()}")
            content = df.to_dict(orient="list")
            logging.debug(f"Converted DataFrame to dict: {list(content.keys())}")
            return {"type": "dataframe", "content": content}
        
        elif filename.endswith('.docx'):
            # Read DOCX file
            doc = docx.Document(document)
            text = "\n".join([para.text for para in doc.paragraphs])
            logging.debug(f"Extracted text from DOCX: {text[:100]}...")  # Log first 100 characters
            return {"type": "text", "content": text}
        
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read image file and perform OCR
            image = Image.open(document)
            text = pytesseract.image_to_string(image)
            logging.debug(f"Extracted text from image: {text[:100]}...")  # Log first 100 characters
            return {"type": "image", "content": text}
        
        elif filename.endswith('.dcm'):
            logging.debug("Detected DICOM file format.")
            return read_medical_image(document)
        
        else:
            logging.error("Unsupported document type.")
            return {"error": "Unsupported document type."}
    
    except Exception as e:
        logging.error(f"Error reading document: {e}")
        return {"error": str(e)}

def analyze_document(content):
    """
    Analyzes the content of a document based on its type.
    """
    logging.debug(f"Analyzing document content of type: {content.get('type')}")
    try:
        if content.get("type") == "text":
            text = content["content"]
            return analyze_text(text)
        elif content.get("type") == "dataframe":
            df = pd.DataFrame(content["content"])
            logging.debug(f"DataFrame for analysis: {df.head()}")
            text = " ".join(df.apply(lambda x: " ".join(x.dropna().astype(str)), axis=1))
            logging.debug(f"Generated text from DataFrame: {text[:100]}...")  # Log first 100 characters
            return {**analyze_text(text), **analyze_dataframe(df)}
        elif content.get("type") == "image":
            text = content["content"]
            return analyze_text(text)
        elif content.get("type") == "medical_image":
            return analyze_medical_image(content)
        else:
            logging.error("Unable to analyze content: Unsupported content type.")
            return {"error": "Unable to analyze content."}
    except Exception as e:
        logging.error(f"Error analyzing document: {e}")
        return {"error": str(e)}

def analyze_text(text):
    """
    Analyzes text content to generate word count, summary, sentiment, and word cloud.
    """
    try:
        word_count = len(text.split())
        truncated_text = text[:MAX_INPUT_LENGTH]
        blob = TextBlob(truncated_text)
        vader_sentiment = analyzer.polarity_scores(truncated_text)
        
        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "positivity": vader_sentiment['pos'],
            "negativity": vader_sentiment['neg'],
            "neutrality": vader_sentiment['neu'],
            "compound": vader_sentiment['compound']
        }

        summary = summarizer(truncated_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        wordcloud = generate_wordcloud(truncated_text)
        
        analysis = {
            "word_count": word_count,
            "summary": summary,
            "sentiment": sentiment,
            "wordcloud": wordcloud
        }
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing text: {e}")
        return {"error": str(e)}

def analyze_dataframe(df):
    """
    Analyzes a dataframe to generate statistical summaries and visualizations.
    """
    try:
        # Using a safe method to convert DataFrame description to dictionary
        description = df.describe(include='all').to_dict()
        logging.debug(f"DataFrame description: {description}")

        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        graphs = {}
        if numerical_columns:
            graphs['histogram'] = generate_histogram(df[numerical_columns])
        if categorical_columns:
            graphs['bar_chart'] = generate_bar_chart(df[categorical_columns])
        if categorical_columns:
            graphs['pie_chart'] = generate_pie_chart(df[categorical_columns])
        
        analysis = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "description": description,
            "column_summary": {col: {"data_type": str(df[col].dtype), "example_values": df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()} for col in df.columns},
            "graphs": graphs
        }
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing dataframe: {e}")
        return {"error": str(e)}

def generate_histogram(df):
    """
    Generates histograms for numerical columns in a dataframe.
    """
    try:
        num_columns = len(df.columns)
        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5*num_columns))
        if num_columns == 1:
            df.hist(ax=axes)
        else:
            for i, col in enumerate(df.columns):
                ax = axes[i]
                df[col].hist(ax=ax)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating histogram: {e}")
        return {"error": str(e)}

def generate_bar_chart(df):
    """
    Generates bar charts for categorical columns in a dataframe.
    """
    try:
        num_columns = len(df.columns)
        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5*num_columns))
        if num_columns == 1:
            df.plot(kind='bar', ax=axes)
        else:
            for i, col in enumerate(df.columns):
                ax = axes[i]
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Chart of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating bar chart: {e}")
        return {"error": str(e)}

def generate_pie_chart(df):
    """
    Generates pie charts for categorical columns in a dataframe.
    """
    try:
        num_columns = len(df.columns)
        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5*num_columns))
        if num_columns == 1:
            df.plot(kind='pie', ax=axes, autopct='%1.1f%%')
        else:
            for i, col in enumerate(df.columns):
                ax = axes[i]
                df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title(f'Pie Chart of {col}')
                ax.set_ylabel('')
        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating pie chart: {e}")
        return {"error": str(e)}

def generate_wordcloud(text):
    """
    Generates a word cloud from text.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        logging.info("Word cloud generated successfully")
        return img_base64
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        return {"error": str(e)}

def read_medical_image(document):
    """
    Reads and extracts information from a DICOM file.
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(document)
        pixel_array = ds.pixel_array

        # Convert to an 8-bit image for better compatibility with pytesseract
        image_8bit = cv2.convertScaleAbs(pixel_array, alpha=(255.0 / pixel_array.max()))

        # Perform OCR on the image
        text = pytesseract.image_to_string(Image.fromarray(image_8bit))

        # Extract metadata
        metadata = {
            "PatientName": str(ds.get("PatientName", "N/A")),
            "StudyDate": str(ds.get("StudyDate", "N/A")),
            "Modality": str(ds.get("Modality", "N/A")),
            "BodyPartExamined": str(ds.get("BodyPartExamined", "N/A"))
        }

        return {"type": "medical_image", "content": text, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error reading medical image: {e}")
        return {"error": str(e)}

def analyze_medical_image(content):
    """
    Analyzes the content and metadata of a medical image.
    """
    try:
        text = content["content"]
        metadata = content["metadata"]

        # Perform any specific medical report analysis (e.g., keywords, diagnoses extraction)
        analysis = analyze_text(text)
        analysis["metadata"] = metadata

        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing medical image: {e}")
        return {"error": str(e)}

# Additional helper functions or updates can be added as needed.
