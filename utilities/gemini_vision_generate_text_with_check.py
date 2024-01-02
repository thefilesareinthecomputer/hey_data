'''this script passes an image to gemini for a description and then passes gemini's output to gemini again 
with a fact-checking prompt before saving and printing both responses'''

# STANDARD IMPORTS ###################################################################################################################################

from datetime import datetime, timedelta
from io import BytesIO, StringIO
from math import radians, cos, sin, asin, sqrt
from urllib.parse import urlparse, urljoin
import asyncio
import base64
import calendar
import datetime
import json
import os
import queue
import re
import ssl
import subprocess
import threading
import time
import tkinter as tk
import traceback
import webbrowser

# THIRD PARTY IMPORTS ###################################################################################################################################

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pyppeteer import launch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import certifi
import google.generativeai as genai
import numpy as np
import pandas as pd
import PIL.Image
import pyautogui
import pytz
import requests
import speech_recognition as sr
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
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')  # Daniel
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
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize helper models
client = OpenAI()
genai.configure(api_key=google_gemini_api_key)
model = genai.GenerativeModel('gemini-pro-vision')

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

# MENU PARSING AGENT THAT USES A SAVED PHOTO ###################################################################################################################################

# STABLE VERSION 

# img = PIL.Image.open(f'{SOURCE_DATA_DIR_PATH}/menu.png')

# response = model.generate_content(["Please list all menu items on this menu, by category, as a data table. Do not hallucinate any false words. You are converting a pdf of a restaurant menu into a data table that can be queried. Ensure the data table is clear and accurate. Keep in mind that menu categories are often the menu headers, and the common categories for menu items include: Salads, Appetizers, Entrees, Desserts, Mains, Sides, Sushi, Soups, Seafood, Sandwiches, Plant-Based, Vegetarian, etc. Keep in mind that menus differ in format. Some sections may have all items listed vertically, whereas some sections may contain multiple columns of menu items. Some menu items may include a description or listed ingredients, and some amy not. Please be discerning and ensure you are recognizing the appropriate entities on the menu as what they really are. Please make sure to gather all of the items on the menu and group them all into the correct categories. Thanks in advance for your hard work. Make sure your output is a data table. Only reply with the data table. Include these columns in the data table: index, menu item, category, price, ingredients.", img])

# time.sleep(1)

# response.resolve()

# time.sleep(1)

# response_1 = response.text

# time.sleep(1)

# response_second_pass = model.generate_content([f"Please look at this data table and this image of a restaurant food menu. A moment ago, an AI model used computer vision to read this food menu into this structured data table, but it was not 100% accurate. Please examine this data table against this food menu and validate the accuracy in the data table. If you discover any incorrect or missing items, please correct them or add them. This was the first pass of the data table, followed by the image. The image is the source of truth. Please ensure the data table reflects the daa in the image in a structured way. Please provide your output as a data table with the same column structure as in the current one. Please make sure to gather all of the items on the menu and group them all into the correct categories. Thanks in advance for your hard work. Make sure your output is a data table. Only reply with the data table. Include these columns in the data table: index, menu item, category, price, ingredients. Here is the data table for you to correct: {response_1}", img])

# time.sleep(1)

# response_second_pass.resolve()

# time.sleep(1)

# response_2 = response_second_pass.text

# time.sleep(1)

# print(f'RESPONSE 1 \n\n {response_1}\n')
# print(f'RESPONSE 2 \n\n {response_2}\n')

# # Convert the content of the response to a .txt file and save it
# with open(f'{FILE_DROP_DIR_PATH}/menu_response_1.txt', 'w') as f:
#     f.write(response_1)

# # Convert the content of the response to a .txt file and save it
# with open(f'{FILE_DROP_DIR_PATH}/menu_response_2.txt', 'w') as f:
#     f.write(response_2)
    
# MENU PARSING AGENT THAT USES A URL PHOTO ###################################################################################################################################

# URL of the image
image_url = 'https://hillstone.com/menus/hillstone/Hillstone%20Park%20Avenue%20South%20Dinner.pdf?version=v-1703997983'  # Replace with your image URL

# Fetch the image from the URL
response = requests.get(image_url, verify=False)
if response.status_code == 200:  # Check if the request was successful
    img = Image.open(BytesIO(response.content))
else:
    print(f"Failed to retrieve image. Status code: {response.status_code}")
    img = None  # Or handle the error as needed

# Use `img` in model.generate_content() call
if img:
    response = model.generate_content(["Please list all menu items on this menu, by category, as a data table. Do not hallucinate any false words. You are converting a pdf of a restaurant menu into a data table that can be queried. Ensure the data table is clear and accurate. Keep in mind that menu categories are often the menu headers, and the common categories for menu items include: Salads, Appetizers, Entrees, Desserts, Mains, Sides, Sushi, Soups, Seafood, Sandwiches, Plant-Based, Vegetarian, etc. Keep in mind that menus differ in format. Some sections may have all items listed vertically, whereas some sections may contain multiple columns of menu items. Some menu items may include a description or listed ingredients, and some amy not. Please be discerning and ensure you are recognizing the appropriate entities on the menu as what they really are. Please make sure to gather all of the items on the menu and group them all into the correct categories. Thanks in advance for your hard work. Make sure your output is a data table. Only reply with the data table. Include these columns in the data table: index, menu item, category, price, ingredients.", img])

    time.sleep(1)

    response.resolve()

    time.sleep(1)

    response_1 = response.text

    time.sleep(1)

    response_second_pass = model.generate_content([f"Please look at this data table and this image of a restaurant food menu. A moment ago, an AI model used computer vision to read this food menu into this structured data table, but it was not 100% accurate. Please examine this data table against this food menu and validate the accuracy in the data table. If you discover any incorrect or missing items, please correct them or add them. This was the first pass of the data table, followed by the image. The image is the source of truth. Please ensure the data table reflects the daa in the image in a structured way. Please provide your output as a data table with the same column structure as in the current one. Please make sure to gather all of the items on the menu and group them all into the correct categories. Thanks in advance for your hard work. Make sure your output is a data table. Only reply with the data table. Include these columns in the data table: index, menu item, category, price, ingredients. Here is the data table for you to correct: {response_1}", img])

    time.sleep(1)

    response_second_pass.resolve()

    time.sleep(1)

    response_2 = response_second_pass.text

    time.sleep(1)

    print(f'RESPONSE 1 \n\n {response_1}\n')
    print(f'RESPONSE 2 \n\n {response_2}\n')

    # Convert the content of the response to a .txt file and save it
    with open(f'{FILE_DROP_DIR_PATH}/menu_response_url_1.txt', 'w') as f:
        f.write(response_1)

    # Convert the content of the response to a .txt file and save it
    with open(f'{FILE_DROP_DIR_PATH}/menu_response_url_2.txt', 'w') as f:
        f.write(response_2)













# photo_url = 'https://hillstone.com/menus/hillstone/Hillstone%20Park%20Avenue%20South%20Dinner.pdf?version=v-1703997983'

    

    
    
    
# # This version is in langchain
# llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# # example
# message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "What's in this image?",
#         },  # You can optionally provide text parts
#         {"type": "image_url", "image_url": "https://hillstone.com/menus/hillstone/Hillstone%20Park%20Avenue%20South%20Dinner.pdf?version=v-1703997983"},
#     ]
# )
# llm.invoke([message])












































# # FUNCTION DEFINITIONS ###################################################################################################################################

# # def is_actively_speaking():
# #     global is_actively_speaking
# #     return is_actively_speaking

# # Parsing and recognizing the user's speech input
# def parse_user_speech():
#     global standby  # Reference the global standby variable
    
#     if standby:
#         print("Standby mode. Not listening.")
#         return None
    
#     listener = sr.Recognizer()
    
#     print('Listening...')
#     try:
#         with sr.Microphone() as source:
#             listener.pause_threshold = 1.5
#             input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)
#         print('Processing...')
#         query = listener.recognize_google(input_speech, language='en_US')
#         print(f'You said: {query}\n')
#         return query

#     except sr.WaitTimeoutError:
#         print('Listening timed out.')
#         if not standby:
#             try:
#                 with sr.Microphone() as source:
#                     input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)
#                     print('Processing...')
#                     query = listener.recognize_google(input_speech, language='en_US')
#                     print(f'You said: {query}\n')
#                     return query
#             except sr.WaitTimeoutError:
#                 print('Second listening attempt timed out.')
#             except sr.UnknownValueError:
#                 print('Second listening attempt resulted in unrecognized speech.')
#         return None

#     except sr.UnknownValueError:
#         print('Speech not recognized.')
#         return None

# # def parse_user_speech():
# #     '''
# #     this version was made to handle a network reset error but it seems to be making the speech weird.
# #     '''
# #     global standby  # Reference the global standby variable
    
# #     if standby:
# #         print("Standby mode. Not listening.")
# #         return None
    
# #     listener = sr.Recognizer()
    
# #     def listen_and_recognize():
# #         try:
# #             with sr.Microphone() as source:
# #                 listener.pause_threshold = 1.5
# #                 input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)
# #             print('Processing...')
# #             return listener.recognize_google(input_speech, language='en_US')
# #         except sr.WaitTimeoutError:
# #             print('Listening timed out.')
# #             return None
# #         except sr.UnknownValueError:
# #             print('Speech not recognized.')
# #             return None

# #     while True:
# #         print('Listening...')
# #         try:
# #             query = listen_and_recognize()
# #             if query is not None:
# #                 print(f'You said: {query}\n')
# #                 return query

# #         except ConnectionResetError:
# #             # Handling the connection reset error
# #             print("Apologies, the connection was reset. Please re-state your last message.")
# #             speak_mainframe("Apologies, the connection was reset. Please re-state your last message.")
# #             # No return here, the loop will continue to attempt to listen again

# # Managing the flow of the speech output queue
# def speech_manager():
#     while True:
#         if not speech_queue.empty():
#             item = speech_queue.get()
#             if item is not None:
#                 text, rate, chunk_size, voice = item
#                 if text:
#                     words = text.split()
#                     chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
#                     for chunk in chunks:
#                         subprocess.call(['say', '-v', voice, '-r', str(rate), chunk])
#                 speech_queue.task_done()
#         time.sleep(0.1)
        
# # Speech output voice settings
# def speak_mainframe(text, rate=195, chunk_size=1000, voice=USER_PREFERRED_VOICE):
#     speech_queue.put((text, rate, chunk_size, voice))

# # # Listen for the user to say "robot, reset robot" to reset the robot
# # def listen_for_reset_command():
# #     global reset_mainframe
# #     listener = sr.Recognizer()
# #     with sr.Microphone() as source:
# #         try:
# #             audio = listener.listen(source, timeout=1, phrase_time_limit=5)
# #             query = listener.recognize_google(audio, language='en_US').lower().split()

# #             if query and query[0] == activation_word and query[1] == 'reset' and query[2] == activation_word:
# #                 reset_mainframe = True
# #         except:
# #             pass

# # Translate a spoken phrase from English to another language by saying "robot, translate to {language}"
# def translate(phrase_to_translate, target_language_name):
#     language_code_mapping = {
#         "en": ["english", "Daniel"],
#         "es": ["spanish", "Paulina"],
#         "fr": ["french", "Amélie"],
#         "de": ["german", "Anna"],
#         "it": ["italian", "Alice"],
#         "ru": ["russian", "Milena"],
#         "ja": ["japanese", "Kyoko"],
#     }

#     source_language = USER_PREFERRED_LANGUAGE  # From .env file
#     target_voice = None

#     # Find the language code and voice that matches the target language name
#     target_language_code = None
#     for code, info in language_code_mapping.items():
#         if target_language_name.lower() == info[0].lower():
#             target_language_code = code
#             target_voice = info[1]
#             break

#     if not target_language_code:
#         return f"Unsupported language: {target_language_name}", USER_PREFERRED_VOICE

#     model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language_code}'
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     batch = tokenizer([phrase_to_translate], return_tensors="pt", padding=True)
#     translated = model.generate(**batch)
#     translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
#     return translation[0], target_voice

# # Gathering a summary of a wikipedia page based on user input
# def wikipedia_summary(query = ''):
#     search_results = wikipedia.search(query)
#     if not search_results:
#         print('No results found.')
#     try:
#         wiki_page = wikipedia.page(search_results[0])
#     except wikipedia.DisambiguationError as e:
#         wiki_page = wikipedia.page(e.options[0])
#     print(wiki_page.title)
#     wiki_summary = str(wiki_page.summary)
#     speak_mainframe(wiki_summary)


# # Querying Wolfram|Alpha based on user input
# def search_wolfram_alpha(query):
#     wolfram_client = wolframalpha.Client(wolfram_app_id)
#     try:
#         response = wolfram_client.query(query)
#         print(f"Response from Wolfram Alpha: {response}")

#         # Check if the query was successfully interpreted
#         if not response['@success']:
#             suggestions = response.get('didyoumeans', {}).get('didyoumean', [])
#             if suggestions:
#                 # Handle multiple suggestions
#                 if isinstance(suggestions, list):
#                     suggestion_texts = [suggestion['#text'] for suggestion in suggestions]
#                 else:
#                     suggestion_texts = [suggestions['#text']]

#                 suggestion_message = " or ".join(suggestion_texts)
#                 speak_mainframe(f"Sorry, I couldn't interpret that query. These are the alternate suggestions: {suggestion_message}.")
#             else:
#                 speak_mainframe('Sorry, I couldn\'t interpret that query. Please try rephrasing it.')

#             return 'Query failed.'

#         relevant_pods_titles = [
#             "Result", "Definition", "Overview", "Summary", "Basic information",
#             "Notable facts", "Basic properties", "Notable properties",
#             "Basic definitions", "Notable definitions", "Basic examples",
#             "Notable examples", "Basic forms", "Notable forms",
#             "Detailed Information", "Graphical Representations", "Historical Data",
#             "Statistical Information", "Comparative Data", "Scientific Data",
#             "Geographical Information", "Cultural Information", "Economic Data",
#             "Mathematical Proofs and Derivations", "Physical Constants",
#             "Measurement Conversions", "Prediction and Forecasting", "Interactive Pods"]

#         # Filtering and summarizing relevant pods
#         answer = []
#         for pod in response.pods:
#             if pod.title in relevant_pods_titles and hasattr(pod, 'text') and pod.text:
#                 answer.append(f"{pod.title}: {pod.text}")

#         # Create a summarized response
#         response_text = ' '.join(answer)
#         if response_text:
#             speak_mainframe(response_text)
#         else:
#             speak_mainframe("I found no information in the specified categories.")

#         # Asking user for interest in other pods
#         for pod in response.pods:
#             if pod.title not in relevant_pods_titles:
#                 speak_mainframe(f"Do you want to hear more about {pod.title}? Say 'yes' or 'no'.")
#                 user_input = parse_user_speech().lower()
#                 if user_input == 'yes' and hasattr(pod, 'text') and pod.text:
#                     speak_mainframe(pod.text)
#                     continue
#                 elif user_input == 'no':
#                     break

#         return response_text

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         speak_mainframe('An error occurred while processing the query.')
#         return f"An error occurred: {e}"

# # Get a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location
# def get_local_four_day_hourly_weather_forecast():
#     appid = f'{open_weather_api_key}'

#     # Fetching coordinates from environment variables
#     lat = USER_SELECTED_HOME_LAT
#     lon = USER_SELECTED_HOME_LON

#     # OpenWeatherMap API endpoint for 4-day hourly forecast
#     url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={appid}"

#     response = requests.get(url)
#     print("Response status:", response.status_code)
#     if response.status_code != 200:
#         return "Failed to retrieve weather data."

#     data = response.json()
#     print("Data received:", data)

#     # Process forecast data
#     forecast = ""
#     timezone = pytz.timezone(USER_SELECTED_TIMEZONE)
#     now = datetime.now(timezone)
#     periods = [(now + timedelta(days=i)).replace(hour=h, minute=0, second=0, microsecond=0) for i in range(4) for h in [6, 12, 18, 0]]

#     for i in range(0, len(periods), 4):
#         day_forecasts = []
#         for j in range(4):
#             start, end = periods[i + j], periods[i + j + 1] if j < 3 else periods[i] + timedelta(days=1)
#             period_forecast = [f for f in data['list'] if start <= datetime.fromtimestamp(f['dt'], tz=timezone) < end]
            
#             if period_forecast:
#                 avg_temp_kelvin = sum(f['main']['temp'] for f in period_forecast) / len(period_forecast)
#                 avg_temp_fahrenheit = (avg_temp_kelvin - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
#                 descriptions = set(f['weather'][0]['description'] for f in period_forecast)
#                 time_label = ["morning", "afternoon", "evening", "night"][j]
#                 day_forecasts.append(f"{time_label}: average temperature {avg_temp_fahrenheit:.1f}°F, conditions: {', '.join(descriptions)}")

#         if day_forecasts:
#             forecast_date = periods[i].strftime('%Y-%m-%d')
#             # Convert forecast_date to weekday format aka "Monday", etc.
#             forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').strftime('%A')
#             forecast += f"\n{forecast_date}: {'; '.join(day_forecasts)}."

#     return forecast

# # def talk_to_chatgpt(query):
# #     assistant = client.beta.assistants.create(
# #         name="Math Tutor",
# #         instructions="You are a personal assistant. Help the user with their Python program.",
# #         tools=[{"type": "code_interpreter"}],
# #         model="gpt-3.5-turbo-instruct"
# #     )
    
# #     thread = client.beta.threads.create()
    
# #     message = client.beta.threads.messages.create(
# #         thread_id=thread.id,
# #         role="user",
# #         content=f"{query}"
# #     )
    
# #     run = client.beta.threads.runs.create(
# #         thread_id=thread.id,
# #         assistant_id=assistant.id,
# #         instructions=f"Please address the user as {USER_PREFERRED_NAME}. The user has a premium account."
# #     )
    
# #     run = client.beta.threads.runs.retrieve(
# #         thread_id=thread.id,
# #         run_id=run.id
# #     )

# #     messages = client.beta.threads.messages.list(
# #         thread_id=thread.id
# #     )
    
# #     print(messages)

# # Conduct research focusing on predetermined reliable science websites (INCOMPLETE / NOT YET FUNCTIONAL)
# def scientific_research(query):
#     science_websites = ['https://www.nih.gov/',
#                        'https://www.semanticscholar.org/',
#                        'https://www.medlineplus.gov/',
#                        'https://www.mayoclinic.org/',
#                        'https://arxiv.org/',
#                        'https://blog.scienceopen.com/',
#                        'https://plos.org/',
#                        'https://osf.io/preprints/psyarxiv',
#                        'https://zenodo.org/',
#                        'https://www.ncbi.nlm.nih.gov/pmc',
#                        'https://www.safemedication.com/',
#                        'https://www.health.harvard.edu/',
#                        'https://pubmed.ncbi.nlm.nih.gov/',
#                        'https://www.nlm.nih.gov/',
#                        'https://clinicaltrials.gov/',
#                        'https://www.amjmed.com/',
#                        'https://www.nejm.org/',
#                        'https://www.science.gov/',
#                        'https://www.science.org/',
#                        'https://www.nature.com/',
#                        'https://scholar.google.com/',
#                        'https://www.cell.com/',
#                        'https://www.mayoclinic.org/', 
#                        'https://www.cdc.gov/', 
#                        'https://www.drugs.com/', 
#                        'https://www.nhs.uk/', 
#                        'https://www.medicinenet.com/', 
#                        'https://www.health.harvard.edu/', 
#                        'https://doaj.org/',
#                        ]
#     research_summary = []
#     # scrape the websites for relevent search results from each site above and append them to the research_summary list
#     for site in science_websites:
#         # scrape the site for search results and append them to the research_summary list in the form of a list of strings
#         pass

# # Conducts various targeted stock market reports such as discounts, recommendations, etc. based on user defined watch list
# class Finance:
#     def __init__(self, user_watch_list):
#         self.user_watch_list = user_watch_list
#         self.stock_data = None

#     # # Fetching data for a single stock symbol
#     # def get_stock_info(self, symbol):
#     #     stock = yf.Ticker(symbol)
#     #     hist = stock.history(period="1d")
#     #     if not hist.empty:
#     #         latest_data = hist.iloc[-1]
#     #         return {
#     #             'symbol': symbol,
#     #             'price': latest_data['Close'],
#     #             'change': latest_data['Close'] - latest_data['Open'],
#     #             'percent_change': ((latest_data['Close'] - latest_data['Open']) / latest_data['Open']) * 100
#     #         }
#     #     else:
#     #         return {'symbol': symbol, 'error': 'No data available'}
    
#     def fetch_all_stock_data(self):
#         try:
#             self.stock_data = yf.download(self.user_watch_list, period="1d")
#         except Exception as e:
#             print(f"Error fetching data: {e}")

#     def get_stock_info(self, symbol):
#         if self.stock_data is None:
#             self.fetch_all_stock_data()

#         if symbol not in self.stock_data.columns.levels[1]:
#             return {'symbol': symbol, 'error': 'No data available'}

#         latest_data = self.stock_data[symbol].iloc[-1]
#         if latest_data.isnull().any():
#             return {'symbol': symbol, 'error': 'No data available'}

#         return {
#             'symbol': symbol,
#             'price': latest_data['Close'],
#             'change': latest_data['Close'] - latest_data['Open'],
#             'percent_change': ((latest_data['Close'] - latest_data['Open']) / latest_data['Open']) * 100
#         }

#     def stock_market_report(self, symbols=None):
#         if symbols is None:
#             symbols = self.user_watch_list
#         stock_data_list = []
#         for symbol in symbols:
#             if symbol and symbol != 'None':
#                 stock_data = self.get_stock_info(symbol)
#                 if 'error' not in stock_data:
#                     stock_data_list.append(stock_data)
#         sorted_stocks = sorted(stock_data_list, key=lambda x: abs(x['percent_change']), reverse=True)
#         significant_changes = [data for data in sorted_stocks if abs(data['percent_change']) > 1]  # Threshold for significant change
#         if not significant_changes:
#             return "Most stocks haven't seen much movement. Here are the ones that are seeing the most action:"
#         report = ["Here are the stocks with the most action:"]
#         for data in significant_changes:
#             change_type = "gained" if data['percent_change'] > 0 else "lost"
#             report_line = f"{data['symbol']} has {change_type} {abs(data['percent_change']):.1f}%\n...\n"
#             report.append(report_line)
#         return '\n'.join(report)

#     def get_industry_avg_pe(self, symbol):
#         industry_average_pe = 25  # Default placeholder value
#         return industry_average_pe

#     def calculate_pe_ratio(self, symbol):
#         stock = yf.Ticker(symbol)
#         pe_ratio = stock.info.get('trailingPE')

#         if pe_ratio is None:  # If trailing P/E is not available, try forward P/E
#             pe_ratio = stock.info.get('forwardPE')

#         return pe_ratio

#     def is_undervalued(self, symbol, pe_ratio):
#         if pe_ratio is None:
#             return False  # If PE ratio data is not available, return False

#         industry_avg_pe = self.get_industry_avg_pe(symbol)
#         return pe_ratio < industry_avg_pe

#     # def calculate_rsi(self, data, window=14):
#     #     """Calculate the relative strength index (RSI) of the stock to assess oversold/overbought conditions."""
#     #     delta = data.diff()
#     #     gain = pd.Series(np.where(delta > 0, delta, 0))
#     #     loss = pd.Series(np.where(delta < 0, -delta, 0))
#     #     avg_gain = gain.rolling(window=window).mean()
#     #     avg_loss = loss.rolling(window=window).mean()
#     #     rs = avg_gain / avg_loss
#     #     rsi = 100 - (100 / (1 + rs))
#     #     return rsi
    
#     def calculate_rsi(self, data, window=14):
#         delta = data.diff()
#         gain = np.where(delta > 0, delta, 0)
#         loss = np.where(delta < 0, -delta, 0)

#         avg_gain = pd.Series(gain).rolling(window=window).mean()
#         avg_loss = pd.Series(loss).rolling(window=window).mean()

#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi

#     def calculate_yearly_change(self, hist, years):
#         """Calculate the percentage change over a specified number of years."""
#         if len(hist) > 0:
#             year_start = hist.iloc[0]['Close']
#             year_end = hist.iloc[-1]['Close']
#             return ((year_end - year_start) / year_start) * 100
#         return 0

#     def calculate_period_change(self, hist, period='1M'):
#         """Calculate the percentage change over a specified recent period."""
#         if len(hist) > 0:
#             period_hist = None
#             if period == '1M':
#                 period_hist = hist.loc[hist.index >= (hist.index.max() - pd.DateOffset(months=1))]
#             elif period == '3M':
#                 period_hist = hist.loc[hist.index >= (hist.index.max() - pd.DateOffset(months=3))]
#             if period_hist is not None and not period_hist.empty:
#                 period_start = period_hist.iloc[0]['Close']
#                 period_end = period_hist.iloc[-1]['Close']
#                 return ((period_end - period_start) / period_start) * 100
#         return 0

#     def is_buy_signal(self, hist, rsi, symbol):
#         year_change = self.calculate_yearly_change(hist, 1)
#         recent_change = self.calculate_period_change(hist, '1M')
#         ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
#         pe_ratio = self.calculate_pe_ratio(symbol)

#         if pe_ratio is not None:
#             undervalued = self.is_undervalued(symbol, pe_ratio)
#         else:
#             undervalued = False  # If PE ratio data is not available, consider it not undervalued

#         if (year_change > 5 and recent_change > 2 and 
#             hist['Close'].iloc[-1] > ma50 and rsi < 70 and undervalued):
#             reasons = []
#             if year_change > 5: reasons.append(f"Yearly growth: {year_change:.1f}%. ...")
#             if recent_change > 2: reasons.append(f"Monthly growth {recent_change:.1f}%. ...")
#             if hist['Close'].iloc[-1] > ma50: reasons.append("Above 50-day average. ...")
#             if rsi < 70: reasons.append(f"RSI: {rsi:.1f}. ...")
#             if undervalued: reasons.append(f"P/E ratio: {pe_ratio:.1f}. ...")
#             return True, " and ".join(reasons)
#         return False, ""

#     def is_sell_signal(self, hist, rsi, symbol):
#         year_change = self.calculate_yearly_change(hist, 1)
#         ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
#         pe_ratio = self.calculate_pe_ratio(symbol)
#         industry_avg_pe = self.get_industry_avg_pe(symbol)

#         # Check if pe_ratio is not None before comparing
#         overvalued = False
#         if pe_ratio is not None:
#             overvalued = pe_ratio > industry_avg_pe * 1.2  # Assuming overvalued if 20% higher than industry average

#         if (year_change < 0 or hist['Close'].iloc[-1] < ma50 or rsi > 70 or overvalued):
#             reasons = []
#             if year_change < 0: reasons.append(f"Yearly loss {year_change:.1f}%. ...")
#             if hist['Close'].iloc[-1] < ma50: reasons.append("Below 50-day average. ...")
#             if rsi > 70: reasons.append(f"RSI: {rsi:.1f}. ...")
#             if overvalued: reasons.append(f"P/E ratio: {pe_ratio:.1f}. ...")
#             return True, " or ".join(reasons)
#         return False, ""

#     def find_stock_recommendations(self):
#         buy_recommendations = []
#         sell_recommendations = []
#         hold_recommendations = []
#         for symbol in self.user_watch_list:
#             if symbol == 'None':
#                 continue
#             stock = yf.Ticker(symbol)
#             hist = stock.history(period="1y")
#             if not hist.empty:
#                 rsi = self.calculate_rsi(hist['Close']).iloc[-1]
#                 # Pass the 'symbol' argument to is_buy_signal and is_sell_signal
#                 buy_signal, buy_reason = self.is_buy_signal(hist, rsi, symbol)
#                 sell_signal, sell_reason = self.is_sell_signal(hist, rsi, symbol)
#                 report_line = f"{symbol}: Recommendation: "
#                 if buy_signal:
#                     buy_recommendations.append(report_line + f"Buy. WHY: ... {buy_reason}\n...\n")
#                 elif sell_signal:
#                     sell_recommendations.append(report_line + f"Sell. WHY: ... {sell_reason}\n...\n")
#                 else:
#                     hold_recommendations.append(report_line + "Hold\n...\n")
#         categorized_recommendations = (
#             ["\nBuy Recommendations:\n"] + buy_recommendations +
#             ["\nSell Recommendations:\n"] + sell_recommendations +
#             ["\nHold Recommendations:\n"] + hold_recommendations
#         )
#         return '\n'.join(categorized_recommendations) if any([buy_recommendations, sell_recommendations, hold_recommendations]) else "No recommendations available."

#     def find_discounted_stocks(self):
#         discounted_stocks_report = []
#         for symbol in self.user_watch_list:
#             if symbol == 'None':  # Skip if the symbol is 'None'
#                 continue
#             stock = yf.Ticker(symbol)
#             hist = stock.history(period="1y")
#             if not hist.empty:
#                 year_start = hist.iloc[0]['Close']  # Yearly change calculation
#                 year_end = hist.iloc[-1]['Close']
#                 year_change = ((year_end - year_start) / year_start) * 100
#                 recent_high = hist['Close'].max()  # Discount from high calculation
#                 current_price = year_end
#                 discount_from_high = ((recent_high - current_price) / recent_high) * 100
#                 ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]  # Moving averages calculation
#                 ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
#                 rsi = self.calculate_rsi(hist['Close']).iloc[-1]  # RSI calculation
#                 # Volume trend (optional)
#                 volume_increase = hist['Volume'].iloc[-1] > hist['Volume'].mean()  # Volume trend (optional)
#                 # Criteria check
#                 if (year_change > 5 and 3 <= discount_from_high <= 25 and
#                     (current_price > ma50 or current_price > ma200) and rsi < 40):
#                     report_line = f"{symbol}: Yearly Change: {year_change:.1f}%, Discount: {discount_from_high:.1f}%\n...\n"
#                     discounted_stocks_report.append(report_line)
#         return '\n'.join(discounted_stocks_report) if discounted_stocks_report else "No discounted stocks found meeting the criteria."

# # MAIN LOOP ###################################################################################################################################

# if __name__ == '__main__':
#     finance = Finance(USER_STOCK_WATCH_LIST)
#     threading.Thread(target=speech_manager, daemon=True).start()
#     speak_mainframe(f'{activation_word} online.')
    
#     # # Initialize and start the GUI
#     # root = tk.Tk()
#     # app = MyGUI(root, speech_queue, is_actively_speaking)
#     # root.mainloop()
    
#     while True:
#         user_input = parse_user_speech()
#         if not user_input:
#             continue

#         query = user_input.lower().split()
#         if not query:
#             continue
        
#         # we should simplify thos to just clear the queue and not reset the robot
#         if reset_mainframe:
#             # Reset the robot and start listening again
#             reset_mainframe = False
#             speak_mainframe(f'{activation_word} reset.')
#             continue
        
#         # Wakes from standby mode if the user says the activation word
#         if standby:
#             if query and query[0] == activation_word:
#                 standby = False
#                 speak_mainframe(f'{activation_word} back online.')
#             continue

#         # reset robot
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'reset' and query[2] == activation_word:
#             speak_mainframe('Heard, resetting.')
#             reset_mainframe = True
#             continue
        
#         # standby
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'standby' and query[2] == 'mode':
#             speak_mainframe('Mainframe going on standby.')
#             standby = True
#             continue

#         # end program
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'terminate' and query[2] == 'program':
#             speak_mainframe('Shutting down.')
#             break
        
#         # talk about yourself
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'talk' and query[2] == 'about':
#             if query[3] == 'yourself':
#                 speak_mainframe('Hi, I\'m Robot. I respond to "Robot" and then your command.')
#                 continue
#             if query[3] == 'what' and query[4] == 'you' and query[5] == 'can' and query[6] == 'do':
#                 speak_mainframe('Hi, I\'m Robot... I can perform tasks from voice commands like saying "google search", \
#                                 ...or "take notes" and "recall notes", ...or "chat gpt" for communicating with chat gpt, ...or \
#                                 "wolfram alpha" for interacting with the Wolfram|Alpha computation engine, ...or "translate file",\
#                                 ...or research assignments, ...and more. ...I\'m here to help you with whatever you need. \
#                                 ...Where would you like to start?')
#                 continue
#             else:
#                 speech = f'I don\'t know much about {query[3:]} yet.'
#                 speak_mainframe(speech)
#                 continue

#         # screenshot the screen
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'screenshot':
#             today = datetime.today().strftime('%Y%m%d %H%M%S')         
#             subprocess.call(['screencapture', 'screenshot.png'])
#             # Save the screenshot to the file drop folder
#             subprocess.call(['mv', 'screenshot.png', f'{FILE_DROP_DIR_PATH}/screenshot_{today}.png'])
#             speak_mainframe('Saved.')
#             continue
            
#         # Note taking
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'take' and query[2] == 'notes':
#             today = datetime.today().strftime('%Y%m%d %H%M%S')         
#             speak_mainframe('OK... What is the subject of the note?')
#             new_note_subject = parse_user_speech().lower().replace(' ', '_')
#             speak_mainframe('Ready.')
#             new_note_text = parse_user_speech().lower()
#             with open(f'{NOTES_DROP_DIR_PATH}/notes_{new_note_subject}.txt', 'a') as f:
#                 f.write(f'{today}, {new_note_text}' + '\n')
#             speak_mainframe('Saved.')
#             continue

#         # recall notes
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'recall' and query[2] == 'notes':
#             subject_pattern = r'notes_(\w+)\.txt'
#             subjects = []
#             for file in os.listdir(NOTES_DROP_DIR_PATH):
#                 if file.startswith('notes_'):
#                     subject = re.search(subject_pattern, file).group(1)
#                     subjects.append(subject)
#             speak_mainframe(f'These subjects are present in your notes: {subjects}. Please specify the subject of the note you want to recall.')
#             desired_subject = parse_user_speech().lower().replace(' ', '_')
#             with open(f'{NOTES_DROP_DIR_PATH}/notes_{desired_subject}.txt', 'r') as f:
#                 note_contents = f.read()
#                 note = []
#                 for row in note_contents.split('\n'):  # Split each row into a list of words and begin speaking on the 3rd word of each row
#                     row = row.split()
#                     if len(row) >= 3:  # Check if there are at least 3 words in the row
#                         note.append(' '.join(row[2:]))
#                 speak_mainframe(f'{note}')
#             continue
        
#         # google search
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'google' and query[2] == 'search':
#             speak_mainframe('Heard')
#             query = ' '.join(query[3:])
#             url = f'https://www.google.com/search?q={query}'
#             webbrowser.open(url, new=1)
#             continue

#         # click
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'click':
#             # Perform a click at the current cursor position
#             pyautogui.click()
#             continue

#         # move up
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'north':
#             direction = query[1]
#             # Remove commas and convert to an integer
#             distance = int(query[2].replace(',', ''))  
#             pyautogui.move(0, -distance, duration=0.1)
#             continue

#         # move down
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'south':
#             direction = query[1]
#             # Remove commas and convert to an integer
#             distance = int(query[2].replace(',', ''))  
#             pyautogui.move(0, distance, duration=0.1) 
#             continue

#         # move left
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'west':
#             direction = query[1]
#             # Remove commas and convert to an integer
#             distance = int(query[2].replace(',', ''))  
#             pyautogui.move(-distance, 0, duration=0.1)
#             continue

#         # move right
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'east':
#             direction = query[1]
#             # Remove commas and convert to an integer
#             distance = int(query[2].replace(',', ''))  
#             pyautogui.move(distance, 0, duration=0.1)
#             continue

#         # translate
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'translate' and query[2] == 'to':
#             target_language_name = query[3]
#             speak_mainframe(f'Speak the phrase you want to translate.')
#             phrase_to_translate = parse_user_speech().lower()
#             translation, target_voice = translate(phrase_to_translate, target_language_name)
#             speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
#             continue

#         # wikipedia summary
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'wiki' and query[2] == 'research':
#             speak_mainframe(f'What would you like summarized from Wikipedia?')
#             wikipedia_summary_query = parse_user_speech().lower()
#             print("Wikipedia Query:", wikipedia_summary_query)  # Debugging print statement
#             speak_mainframe(f'Searching {wikipedia_summary_query}')
#             summary = wikipedia_summary(wikipedia_summary_query)  # Get the summary
#             speak_mainframe(summary)  # Speak the summary
#             continue

#         # youtube video
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'youtube' and query[2] == 'video':
#             speak_mainframe(f'What would you like to search for on YouTube?')
#             youtube_query = parse_user_speech().lower()
#             print("YouTube Query:", youtube_query)  # Debugging print statement
#             speak_mainframe(f'Searching YouTube for {youtube_query}')
#             url = f'https://www.youtube.com/results?search_query={youtube_query}'
#             webbrowser.open(url)
#             continue
        
#         # wolfram alpha
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'computation' and query[2] == 'engine':
#             speak_mainframe(f'What would you like to ask?')
#             wolfram_alpha_query = parse_user_speech().lower()
#             speak_mainframe(f'Weird question but ok...')
#             result = search_wolfram_alpha(wolfram_alpha_query)
#             print(f'User: {wolfram_alpha_query} \nWolfram|Alpha: {result}')
#             continue
                
#         # open weather forecast
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'weather' and query[2] == 'forecast':
#             speak_mainframe(f'OK - beginning weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}')
#             weather_forecast = get_local_four_day_hourly_weather_forecast()
#             print("Weather forecast:", weather_forecast)
#             speak_mainframe(f'Weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}: {weather_forecast}')
#             continue

#         # stock market reports
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'stock' and query[2] == 'report':
#             today = datetime.today().strftime('%Y%m%d %H%M%S')         
#             speak_mainframe('Category?')
#             focus = parse_user_speech().lower()
#             if focus == 'discounts':
#                 discounts_update = finance.find_discounted_stocks()
#                 speak_mainframe(discounts_update)
#                 print(discounts_update)
#                 continue
#             if focus == 'recommendations':
#                 recs_update = finance.find_stock_recommendations()
#                 speak_mainframe(recs_update)
#                 print(recs_update)
#                 continue
#             if focus == 'yesterday':
#                 portfolio_update = finance.stock_market_report()
#                 speak_mainframe(portfolio_update)
#                 print(portfolio_update)
#                 continue
#             if focus == 'world':
#                 world_market_update = finance.stock_market_report(['^GSPC', '^IXIC', '^DJI'])  # S&P 500, NASDAQ, Dow Jones
#                 speak_mainframe(world_market_update)
#                 print(world_market_update)
#                 continue
#             if focus == 'single':
#                 speak_mainframe('Please spell the stock ticker, letter by letter.')
#                 attempts = 0
#                 while attempts < 3:  # Limit the number of attempts
#                     raw_stock_input = parse_user_speech().upper().strip()
#                     stock = ''.join(raw_stock_input.split())  # Simplify the handling of the input
#                     speak_mainframe(f"Did you say {stock}?")
#                     confirmation = parse_user_speech().lower()
#                     if 'yes' in confirmation:
#                         stock_info = finance.get_stock_info(stock)
#                         if 'error' in stock_info:
#                             speak_mainframe(f"Error fetching data for {stock}.")
#                         else:
#                             hist = yf.Ticker(stock).history(period="1y")
#                             if not hist.empty:
#                                 rsi = finance.calculate_rsi(hist['Close']).iloc[-1]
#                                 buy_signal, buy_reason = finance.is_buy_signal(hist, rsi)
#                                 sell_signal, sell_reason = finance.is_sell_signal(hist, rsi)

#                                 # Ensure all numeric outputs have a maximum of one decimal place
#                                 stock_update = (f"Stock: {stock_info['symbol']}\n...\n"
#                                                 f"Price: {stock_info['price']:.1f}\n...\n"
#                                                 f"Change vs LY: {stock_info['change']:.1f}\n...\n"
#                                                 f"Percent Change: {stock_info['percent_change']:.1f}%\n...\n"
#                                                 f"RSI: {rsi:.1f}\n...\n"
#                                                 f"Buy Signal: {'Yes' if buy_signal else 'No'} ({buy_reason})\n...\n"
#                                                 f"Sell Signal: {'Yes' if sell_signal else 'No'} ({sell_reason})\n...\n")
#                                 speak_mainframe(stock_update)
#                                 print(stock_update)
#                             else:
#                                 speak_mainframe(f"No historical data available for {stock}.")
#                         break
#                     else:
#                         speak_mainframe("I didn't catch that. Could you please spell out the stock ticker again?")
#                         attempts += 1
#                 if attempts == 3:
#                     speak_mainframe("Having trouble recognizing the stock ticker. Let's try something else.")
#                 continue
        
#         # Gemini chat
#         if len(query) > 1 and query[0] == activation_word and query[1] == 'call' and query[2] == 'gemini':
#             speak_mainframe('Starting Gemini chat.')
#             chat = model.start_chat(history=[])

#             while True:
#                 user_input = parse_user_speech()
#                 if not user_input:
#                     continue

#                 query = user_input.lower().split()
#                 if not query:
#                     continue

#                 if query[0] == activation_word and query[1] == 'terminate' and query[2] == 'chat':
#                     speak_mainframe('Ending Gemini chat.')
#                     break
#                 else:
#                     response = chat.send_message(f"{user_input}", stream=True)
#                     if response:  
#                         for chunk in response:
#                             speak_mainframe(chunk.text)
                            
#         # # chat gpt command - we should change this to enter a stateful loop with a break command when the user wants to talk to chatgpt
#         # if len(query) > 1 and query[0] == activation_word and query[1] == 'chat' and query[2] == 'gpt':
#         #     speak_mainframe('Initializing ChatGPT. Say "end chat" to exit.')
#         #     chatgpt = client

#         #     while True:
#         #         speak_mainframe('I am listening...')
#         #         chat_query = parse_user_speech().lower()

#         #         # Check if 'end' and 'chat' are consecutive words in the query
#         #         if 'end' in chat_query and 'chat' in chat_query and chat_query.index('end') + 1 == chat_query.index('chat'):
#         #             speak_mainframe('Ending ChatGPT session.')
#         #             break

#         #         chat_response = chatgpt.ask_chatgpt(' '.join(chat_query))
#         #         print(f'ChatGPT: {chat_response}')
#         #         speak_mainframe(chat_response)
                
#         #     continue

#         # # scientific research command
#         # if len(query) > 1 and query[0] == activation_word and query[1] == 'scientific' and query[2] == 'research':
#         #     speak_mainframe(f'What would you like to research?')
#         #     research_query = parse_user_speech().lower()
#         #     print("Research Query:", research_query)  # Debugging print statement
#         #     speak_mainframe(f'Searching {research_query}')
#         #     research = scientific_research(research_query)  # Get the summary
#         #     speak_mainframe(research)  # Speak the summary
#         #     continue
            
        
# # END ###################################################################################################################################


