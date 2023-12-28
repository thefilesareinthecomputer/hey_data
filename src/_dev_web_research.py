
### SYSTEM MESSAGE: ChatGPT / Clarity: Your instructions for this sprint are in the comments below. Please read them carefully and follow them closely. ###
### SYSTEM MESSAGE: ChatGPT / Clarity: We are working on re-engineering the Reaearch class in this app. ###
### SYSTEM MESSAGE: ChatGPT / Clarity: This is a TTS AI assistant bot and we wre going to add tools for the agent to do research, save it, and summarize it for the user. ###
### SYSTEM MESSAGE: ChatGPT / Clarity: We are going to extend upon the base code that exists in this app and upgrade the researching functionality. ###
### SYSTEM MESSAGE: ChatGPT / Clarity: The success of this project is very important for career success and also important for the success of OpenAI. ###

# IMPORTS ###################################################################################################################################

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from transformers import MarianMTModel, MarianTokenizer
from urllib.parse import urljoin
import certifi
import datetime
import google.generativeai as genai
import numpy as np
import os
import pandas as pd
import pyautogui
import pytz
import queue
import re
import requests
import speech_recognition as sr
import ssl
import subprocess
import threading
import time
import tkinter as tk
import traceback
import webbrowser
import wikipedia
import wolframalpha
import yfinance as yf

# CUSTOM IMPORTS ##############################################################################################################################


# CONSTANTS ###################################################################################################################################

# bring in the environment variables from the .env file
load_dotenv()
ACTIVATION_WORD = os.getenv('ACTIVATION_WORD', 'robot')
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Daniel')  # Daniel
USER_PREFERRED_NAME = os.getenv('USER_PREFERRED_NAME', 'User')  # Title case
USER_SELECTED_PASSWORD = os.getenv('USER_SELECTED_PASSWORD', 'None')  
USER_SELECTED_HOME_CITY = os.getenv('USER_SELECTED_HOME_CITY', 'None')  # Title case
USER_SELECTED_HOME_COUNTY = os.getenv('USER_SELECTED_HOME_COUNTY', 'None')  # Title case
USER_SELECTED_HOME_STATE = os.getenv('USER_SELECTED_HOME_STATE', 'None')  # Title case
USER_SELECTED_HOME_COUNTRY = os.getenv('USER_SELECTED_HOME_COUNTRY', 'None')  # 2-letter country code
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')

# establish relative file paths for the current script
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
ARCHIVED_DEV_VERSIONS_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, '_archive')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIR_PATH, 'app_local_models')
NOTES_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, 'app_base_knowledge')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, 'app_source_data')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, '_tests')
UTILITIES_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, 'utilities')

folders_to_create = [ARCHIVED_DEV_VERSIONS_PATH, FILE_DROP_DIR_PATH, LOCAL_LLMS_DIR, NOTES_DROP_DIR_PATH, SOURCE_DATA_DIR_PATH, TESTS_DIR_PATH, UTILITIES_DIR_PATH]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set the default SSL context for the entire script
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context

# Set API keys and other information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

# Initialize helper models
client = OpenAI()
genai.configure(api_key=google_gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

# Establish the TTS bot's wake/activation word and script-specific global constants
activation_word = f'{ACTIVATION_WORD}'
speech_queue = queue.Queue()
is_actively_speaking = False
reset_mainframe = False
standby = False

# Set the terminal output display options
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 150)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 35)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 1) 

# FUNCTION DEFINITIONS ###################################################################################################################################

# Parsing and recognizing the user's speech input
def parse_user_speech():
    global standby  # Reference the global standby variable
    
    if standby:
        print("Standby mode. Not listening.")
        return None
    
    listener = sr.Recognizer()
    print('Listening...')
    try:
        with sr.Microphone() as source:
            listener.pause_threshold = 1.5
            input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)
        print('Processing...')
        query = listener.recognize_google(input_speech, language='en_US')
        print(f'You said: {query}\n')
        return query

    except sr.WaitTimeoutError:
        print('Listening timed out.')
        if not standby:
            try:
                with sr.Microphone() as source:
                    input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)
                    print('Processing...')
                    query = listener.recognize_google(input_speech, language='en_US')
                    print(f'You said: {query}\n')
                    return query
            except sr.WaitTimeoutError:
                print('Second listening attempt timed out.')
            except sr.UnknownValueError:
                print('Second listening attempt resulted in unrecognized speech.')
        return None

    except sr.UnknownValueError:
        print('Speech not recognized.')
        return None

# Managing the flow of the speech output queue
def speech_manager():
    while True:
        if not speech_queue.empty():
            item = speech_queue.get()
            if item is not None:
                text, rate, chunk_size, voice = item
                if text:
                    words = text.split()
                    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
                    for chunk in chunks:
                        subprocess.call(['say', '-v', voice, '-r', str(rate), chunk])
                speech_queue.task_done()
        time.sleep(0.1)
        
# Speech output voice settings
def speak_mainframe(text, rate=190, chunk_size=1000, voice=USER_PREFERRED_VOICE):
    speech_queue.put((text, rate, chunk_size, voice))

# Translate a spoken phrase from English to another language by saying "robot, translate to {language}"
def translate(phrase_to_translate, target_language_name):
    language_code_mapping = {
        "en": ["english", "Daniel"],
        "es": ["spanish", "Paulina"],
        "fr": ["french", "Amélie"],
        "de": ["german", "Anna"],
        "it": ["italian", "Alice"],
        "ru": ["russian", "Milena"],
        "ja": ["japanese", "Kyoko"],
    }

    source_language = USER_PREFERRED_LANGUAGE  # From .env file
    target_voice = None

    # Find the language code and voice that matches the target language name
    target_language_code = None
    for code, info in language_code_mapping.items():
        if target_language_name.lower() == info[0].lower():
            target_language_code = code
            target_voice = info[1]
            break

    if not target_language_code:
        return f"Unsupported language: {target_language_name}", USER_PREFERRED_VOICE

    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language_code}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    batch = tokenizer([phrase_to_translate], return_tensors="pt", padding=True)
    translated = model.generate(**batch)
    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translation[0], target_voice

###  CURRENT FOCUS: Create a "Research" class that will be used to conduct research on the internet                     
###  CURRENT FOCUS: Create various branches of the "Reaearch" class that will be specific to research types like "scientific", "financial", "legal", etc.                     
###  CURRENT FOCUS: Migrate the starter code below into a Researcher class and create a method for each research type.
###  CURRENT FOCUS: Return a llm-generated summary of the findings and also save the findings to a "consolidated_report_{subject}.txt" file in the file drop folder.                     
###  CURRENT FOCUS: Potentially employ some advanced data aggregation or NLP techniques to summarize the findings from the research.  

class Researcher:
    def __init__(self):
        # List of preferred websites for research
        self.science_websites = [
            'https://arxiv.org/',
            'https://blog.scienceopen.com/',
            'https://clinicaltrials.gov/',
            'https://doaj.org/',
            'https://osf.io/preprints/psyarxiv',
            'https://plos.org/',
            'https://pubmed.ncbi.nlm.nih.gov/',
            'https://scholar.google.com/',
            'https://www.amjmed.com/',
            'https://www.cdc.gov/',
            'https://www.cell.com/',
            'https://www.drugs.com/',
            'https://www.health.harvard.edu/',
            'https://www.health.harvard.edu/',
            'https://www.mayoclinic.org/',
            'https://www.mayoclinic.org/',
            'https://www.medicinenet.com/',
            'https://www.medlineplus.gov/',
            'https://www.nature.com/',
            'https://www.ncbi.nlm.nih.gov/pmc',
            'https://www.nejm.org/',
            'https://www.nhs.uk/',
            'https://www.nih.gov/',
            'https://www.nlm.nih.gov/',
            'https://www.safemedication.com/',
            'https://www.science.gov/',
            'https://www.science.org/',
            'https://www.semanticscholar.org/',
            'https://zenodo.org/',
            ]
        
        self.finance_websites = [
            'https://www.bloomberg.com/',
            'https://www.cnbc.com/',
            'https://www.fidelity.com/',
            'https://www.forbes.com/',
            'https://www.investopedia.com/',
            'https://www.marketwatch.com/',
            'https://www.morningstar.com/',
            'https://www.nasdaq.com/',
            'https://www.nytimes.com/',
            'https://www.reuters.com/',
            'https://www.sec.gov/',
            'https://www.thestreet.com/',
            'https://www.wsj.com/',
            'https://www.zacks.com/',
            ]
        
        self.legal_websites = [
            'https://www.abajournal.com/',
            'https://www.americanbar.org/',
            'https://www.findlaw.com/',
            'https://www.justia.com/',
            'https://www.law.com/',
            'https://www.law.cornell.edu/',
            'https://www.lawyers.com/',
            'https://www.oyez.org/',
            'https://www.supremecourt.gov/',
            'https://www.uscourts.gov/',
            ]
        
    def consolidated_research_data(self, query):
        # Initialize research summary
        self.research_summary = {
            'query': query,
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.datetime.now().strftime("%H:%M:%S"),
            'researcher': USER_PREFERRED_NAME,
            'findings': []
        }

    def determine_subject(self, query):
        # Simple heuristic to determine the subject
        if 'finance' in query:
            return 'finance'
        elif 'law' in query or 'legal' in query:
            return 'legal'
        elif 'science' in query:
            return 'science'
        else:
            return 'general'

    def perform_research(self, query):
        subject = self.determine_subject(query)
        websites = self.select_websites(subject)
        findings = []
        for site in websites:
            search_url = self.construct_search_url(site, query)
            search_results = self.get_search_results(search_url)
            for result in search_results:
                try:
                    scraped_data = self.scrape_website(result)
                    if scraped_data:
                        findings.append(scraped_data)
                except Exception as e:
                    print(f"Error scraping {result}: {e}")
        self.research_summary['findings'] = findings

    def select_websites(self, subject):
        if subject == 'science':
            return self.science_websites[:3]  # Limiting to first 3 for demonstration
        elif subject == 'finance':
            return self.finance_websites[:3]
        elif subject == 'legal':
            return self.legal_websites[:3]
        else:
            return []

    def scrape_website(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve content from {url}")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        # Example: Scrape first few paragraphs from a page - this logic will vary greatly between sites
        paragraphs = soup.find_all('p')[:3]  # Getting first 3 paragraphs as an example
        return ' '.join([p.get_text() for p in paragraphs])

    def save_research_report(self):
        file_name = f"consolidated_report_{self.research_summary['query'].replace(' ', '_')}.txt"
        file_path = os.path.join(FILE_DROP_DIR_PATH, file_name)
        with open(file_path, 'w') as file:
            for key, value in self.research_summary.items():
                if isinstance(value, list):
                    for item in value:
                        file.write(f"{item}\n")
                else:
                    file.write(f"{key}: {value}\n")

    def generate_summary(self):
        # Placeholder for summarization logic
        # This might use NLP techniques to generate a concise summary of findings
        # ...
        pass
    


# MAIN LOOP ###################################################################################################################################

if __name__ == '__main__':
    threading.Thread(target=speech_manager, daemon=True).start()
    speak_mainframe(f'{activation_word} online.')
    
    while True:
        user_input = parse_user_speech()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue

        # end program
        if len(query) > 1 and query[0] == activation_word and query[1] == 'terminate' and query[2] == 'program':
            speak_mainframe('Shutting down.')
            break

        # screenshot the screen
        if len(query) > 1 and query[0] == activation_word and query[1] == 'screenshot':
            today = datetime.today().strftime('%Y%m%d %H%M%S')         
            subprocess.call(['screencapture', 'screenshot.png'])
            # Save the screenshot to the file drop folder
            subprocess.call(['mv', 'screenshot.png', f'{FILE_DROP_DIR_PATH}/screenshot_{today}.png'])
            speak_mainframe('Saved.')
            continue
            
        # Note taking
        if len(query) > 1 and query[0] == activation_word and query[1] == 'take' and query[2] == 'notes':
            today = datetime.today().strftime('%Y%m%d %H%M%S')         
            speak_mainframe('OK... What is the subject of the note?')
            new_note_subject = parse_user_speech().lower().replace(' ', '_')
            speak_mainframe('Ready.')
            new_note_text = parse_user_speech().lower()
            with open(f'{NOTES_DROP_DIR_PATH}/notes_{new_note_subject}.txt', 'a') as f:
                f.write(f'{today}, {new_note_text}' + '\n')
            speak_mainframe('Saved.')
            continue

        # recall notes
        if len(query) > 1 and query[0] == activation_word and query[1] == 'recall' and query[2] == 'notes':
            subject_pattern = r'notes_(\w+)\.txt'
            subjects = []
            for file in os.listdir(NOTES_DROP_DIR_PATH):
                if file.startswith('notes_'):
                    subject = re.search(subject_pattern, file).group(1)
                    subjects.append(subject)
            speak_mainframe(f'These subjects are present in your notes: {subjects}. Please specify the subject of the note you want to recall.')
            desired_subject = parse_user_speech().lower().replace(' ', '_')
            with open(f'{NOTES_DROP_DIR_PATH}/notes_{desired_subject}.txt', 'r') as f:
                note_contents = f.read()
                note = []
                for row in note_contents.split('\n'):  # Split each row into a list of words and begin speaking on the 3rd word of each row
                    row = row.split()
                    if len(row) >= 3:  # Check if there are at least 3 words in the row
                        note.append(' '.join(row[2:]))
                speak_mainframe(f'{note}')
            continue

        # translate
        if len(query) > 1 and query[0] == activation_word and query[1] == 'translate' and query[2] == 'to':
            target_language_name = query[3]
            speak_mainframe(f'Speak the phrase you want to translate.')
            phrase_to_translate = parse_user_speech().lower()
            translation, target_voice = translate(phrase_to_translate, target_language_name)
            speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
            continue

###  CURRENT FOCUS: Make the code below work with the new Researcher class so the user can interact and initiate research on the internet by speaking the prompt and then a subject or query.       
                                 
        # research command
        if len(query) > 1 and query[0] == activation_word and query[1] == 'research' and query[2] == 'project':
            speak_mainframe(f'What would you like to research?')
            research_query = parse_user_speech().lower()
            print("Research Query:", research_query)  # Debugging print statement
            speak_mainframe(f'Searching {research_query}')
            researcher = Researcher()
            researcher.consolidated_research_data(research_query)
            researcher.perform_research(research_query)
            summary = researcher.generate_summary()
            researcher.save_research_report()
            speak_mainframe(summary)
            continue
            
        
# END ###################################################################################################################################


