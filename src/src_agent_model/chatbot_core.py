import sqlite3
import json
import nltk
import os
import ssl
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure SSL context is properly set for downloads
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

# Check if nltk packages are downloaded, download if not
def download_nltk_package(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

download_nltk_package('punkt')
download_nltk_package('wordnet')

class NLPUtils:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def lemmatize(self, text):
        words = nltk.word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    def vectorize(self, text):
        return self.vectorizer.transform([text])

class ChatbotUtils:
    def __init__(self, db_path, response_templates_path):
        self.nlp_utils = NLPUtils()
        self.db_manager = DatabaseManager(db_path)
        self.response_templates = self.load_response_templates(response_templates_path)

    def load_response_templates(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def get_response(self, user_input):
        processed_input = self.nlp_utils.lemmatize(user_input)
        response = self.generate_response(processed_input)
        self.db_manager.log_conversation(user_input, response)
        return response

    def generate_response(self, processed_input):
        # Placeholder - implement response generation logic
        # This can include looking up the response_templates
        return "Processed response for: " + processed_input

    def run_standalone(self):
        print("Chatbot running in standalone mode. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.get_response(user_input)
            print("Bot:", response)

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                bot_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        self.conn.commit()

    def log_conversation(self, user_input, bot_response, bot_name):
        cursor = self.conn.cursor()
        timestamp = datetime.now()  # Get current date and time
        query = "INSERT INTO conversations (user_input, bot_response, bot_name, timestamp) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (user_input, bot_response, bot_name, timestamp))
        self.conn.commit()
        
if __name__ == "__main__":
    db_path = 'your_database_path.db'
    response_templates_path = 'path_to_responses.json'
    chatbot = ChatbotUtils(db_path, response_templates_path)
    chatbot.run_standalone()





"""



## this worked for importing into the main module:

from src_agent.chatbot_core import NLPUtils, ChatbotUtils, DatabaseManager




## now i need to really make these work:

import sqlite3
import json
import nltk
import os
import ssl
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure SSL context is properly set for downloads
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

# Check if nltk packages are downloaded, download if not
def download_nltk_package(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

download_nltk_package('punkt')
download_nltk_package('wordnet')

class NLPUtils:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def lemmatize(self, text):
        words = nltk.word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    def vectorize(self, text):
        return self.vectorizer.transform([text])

class ChatbotUtils:
    def __init__(self, db_path, response_templates_path):
        self.nlp_utils = NLPUtils()
        self.db_manager = DatabaseManager(db_path)
        self.response_templates = self.load_response_templates(response_templates_path)

    def load_response_templates(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def get_response(self, user_input):
        processed_input = self.nlp_utils.lemmatize(user_input)
        response = self.generate_response(processed_input)
        self.db_manager.log_conversation(user_input, response)
        return response

    def generate_response(self, processed_input):
        # Placeholder - implement response generation logic
        # This can include looking up the response_templates
        return "Processed response for: " + processed_input

    def run_standalone(self):
        print("Chatbot running in standalone mode. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.get_response(user_input)
            print("Bot:", response)

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                bot_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        self.conn.commit()

    def log_conversation(self, user_input, bot_response, bot_name):
        cursor = self.conn.cursor()
        timestamp = datetime.now()  # Get current date and time
        query = "INSERT INTO conversations (user_input, bot_response, bot_name, timestamp) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (user_input, bot_response, bot_name, timestamp))
        self.conn.commit()
        
if __name__ == "__main__":
    db_path = 'your_database_path.db'
    response_templates_path = 'path_to_responses.json'
    chatbot = ChatbotUtils(db_path, response_templates_path)
    chatbot.run_standalone()















"""