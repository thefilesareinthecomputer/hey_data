
# STANDARD IMPORTS ###################################################################################################################################

from datetime import datetime, timedelta
from difflib import get_close_matches
from io import BytesIO, StringIO
from math import radians, cos, sin, asin, sqrt
from urllib.parse import urlparse, urljoin
import asyncio
import base64
import calendar
import json
import os
import pickle
import queue
import random
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
from nltk.stem import WordNetLemmatizer
from PIL import Image
from pyppeteer import launch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import certifi
import google.generativeai as genai
import nltk
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

# CUSTOM MODULE IMPORTS ##############################################################################################################################

# CONSTANTS ###################################################################################################################################

# Load environment variables and verify the supporting directories exist
load_dotenv()
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')  # Daniel
USER_PREFERRED_NAME = os.getenv('USER_PREFERRED_NAME', 'User')  # Title case
USER_SELECTED_HOME_CITY = os.getenv('USER_SELECTED_HOME_CITY', 'None')  # Title case
USER_SELECTED_HOME_COUNTY = os.getenv('USER_SELECTED_HOME_COUNTY', 'None')  # Title case
USER_SELECTED_HOME_STATE = os.getenv('USER_SELECTED_HOME_STATE', 'None')  # Title case
USER_SELECTED_HOME_COUNTRY = os.getenv('USER_SELECTED_HOME_COUNTRY', 'None')  # 2-letter country code
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
USER_STOCK_WATCH_LIST = os.getenv('USER_STOCK_WATCH_LIST', 'None').split(',')  # Comma separated list of stock symbols
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
ARCHIVED_DEV_VERSIONS_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_archive')
DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')
folders_to_create = [ARCHIVED_DEV_VERSIONS_PATH, DATABASES_DIR_PATH, FILE_DROP_DIR_PATH, LOCAL_LLMS_DIR, BASE_KNOWLEDGE_DIR_PATH, SECRETS_DIR_PATH, SOURCE_DATA_DIR_PATH, SRC_DIR_PATH, TESTS_DIR_PATH, UTILITIES_DIR_PATH]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set the default SSL context for the entire script
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')

# Establish the TTS bot's wake/activation word and script-specific global constants
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
speech_queue = queue.Queue()

# Initialize the Google Gemini LLM
genai.configure(api_key=google_gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-pro')

# CLASS DEFINITIONS ###################################################################################################################################

# Main class for custom search engines in Google Cloud, etc. and a static method which engages a chatbot wrapper around the engines
class CustomSearchEngines:
    def __init__(self):
        self.engines = {
            'documentation': {
                'api_key': google_cloud_api_key,
                'cse_id': google_documentation_search_engine_id
            }
            # Add more search engines here if needed
        }

    def search(self, engine_name, query):
        if engine_name not in self.engines:
            print(f"Search engine {engine_name} not found.")
            return None

        engine = self.engines[engine_name]
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': engine['api_key'],
            'cx': engine['cse_id'],
            'q': query
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error in {engine_name} Search: {e}")
            return None

    @staticmethod
    def search_chat_bot():
        search_engines = CustomSearchEngines()
        engine_name = ''
        search_results = [] 
        chat = gemini_model.start_chat(history=[])
        speak_mainframe("Specify the search engine to use.")
        while not engine_name:
            input = parse_user_speech()
            if input:
                engine_name = input.lower()
                print(f"Search Assistant Active using {engine_name}.")
                time.sleep(.5)
            if not input:
                continue
            else:
                continue
                            
        speak_mainframe("Please say what you'd like to search")
        
        while True:
            user_input = parse_user_speech()
            
            if not user_input:
                continue
            
            query = user_input.lower()
            
            if query == 'exit search':
                break
            
            results = search_engines.search(engine_name, query)
            
            if results and results.get('items'):
                for item in results['items']:
                    search_results.append(item)  # Storing the entire item
                    print(f"RESULT: {item}\n\n")

                prompt_template = chat.send_message(
                    '''# System Message # - Gemini, you are in a verbal chat with the user via a 
                    STT / TTS application. Please generate your text in a way that sounds like natural speech 
                    when it's spoken by the TTS app. Please avoid monologuing or including anything in the output that will 
                    not sound like natural spoken language. After confirming you understand this message, the chat will proceed. Please 
                    confirm your understanding of these instructions by simply saying "Chat loop is open."''', 
                    stream=True)

                if prompt_template:
                    for chunk in prompt_template:
                        speak_mainframe(chunk.text)
                        time.sleep(.25)
                    time.sleep(1)
                    
                search_analysis = chat.send_message(
                    f'''Hi Gemini. The user just engaged a Google custom 
                    search engine with this query: "{query}". 
                    These are the search results: {search_results}. 
                    Please analyze the results while interpreting the true meaning of 
                    the user's query, then also apply your own internal knowledge. 
                    Please guide the user in how to solve the problem or answer the question in their query. 
                    If necessary, also guide the user in crafting a more efficient and effective query. 
                    Please help guide the user in the right direction. 
                    Your output should be suitable for a verbal chat TTS app that sounds like natural spoken language. 
                    Please keep your answers short, direct, and concise. Thank you!''', 
                    stream=True)
                
                if search_analysis:
                    for chunk in search_analysis:
                        speak_mainframe(chunk.text)
                        time.sleep(.25)
                    time.sleep(1)
                    
                while True:
                    user_input = parse_user_speech()
                    
                    if not user_input:
                        continue
                    
                    query = user_input.lower().split()
                    
                    if query[0] == activation_word and query[1] == 'new' and query[2] == 'search':
                        speak_mainframe("Please say what you'd like to search.")
                        time.sleep(1)
                        break
                    else:
                        response = chat.send_message(f'{user_input}', stream=True)
                        if response:
                            for chunk in response:
                                speak_mainframe(chunk.text)
                            time.sleep(1)
                                
            else:
                speak_mainframe("No results found or an error occurred.")
        
# Conducts various targeted stock market reports such as discounts, recommendations, etc. based on user defined watch list
class StockReports:
    def __init__(self, user_watch_list):
        self.user_watch_list = user_watch_list
        self.stock_data = None
    
    def handle_single_stock_query(self, stock):
        try:
            # Fetch data directly for the requested stock
            stock_data = yf.Ticker(stock)

            # Check if the ticker is valid and has historical data
            hist = stock_data.history(period="1y")
            if hist.empty:
                raise ValueError(f"No historical data available for {stock}")

            rsi = self.calculate_rsi(hist['Close']).iloc[-1]
            buy_signal, buy_reason = self.is_buy_signal(hist, rsi, stock)
            sell_signal, sell_reason = self.is_sell_signal(hist, rsi, stock)

            stock_info = {
                'symbol': stock,
                'price': hist['Close'].iloc[-1],
                'change': hist['Close'].iloc[-1] - hist['Open'].iloc[-1],
                'percent_change': ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
            }

            stock_update = (f"Stock: {stock_info['symbol']}\n...\n"
                            f"Price: {stock_info['price']:.1f}\n...\n"
                            f"Change vs LY: {stock_info['change']:.1f}\n...\n"
                            f"Percent Change: {stock_info['percent_change']:.1f}%\n...\n"
                            f"RSI: {rsi:.1f}\n...\n"
                            f"Buy Signal: {'Yes' if buy_signal else 'No'} ({buy_reason})\n...\n"
                            f"Sell Signal: {'Yes' if sell_signal else 'No'} ({sell_reason})\n...\n")
            speak_mainframe(stock_update)
            print(stock_update)
        except ValueError as e:
            speak_mainframe(str(e))
        except Exception as e:
            speak_mainframe(f"Error fetching data for {stock}: {e}")
                
    def fetch_all_stock_data(self):
        try:
            self.stock_data = yf.download(self.user_watch_list, period="1d")
        except Exception as e:
            print(f"Error fetching data: {e}")

    def get_stock_info(self, symbol):
        if self.stock_data is None:
            self.fetch_all_stock_data()

        if symbol not in self.stock_data.columns.levels[1]:
            return {'symbol': symbol, 'error': 'No data available'}

        latest_data = self.stock_data[symbol].iloc[-1]
        if latest_data.isnull().any():
            return {'symbol': symbol, 'error': 'No data available'}

        return {
            'symbol': symbol,
            'price': latest_data['Close'],
            'change': latest_data['Close'] - latest_data['Open'],
            'percent_change': ((latest_data['Close'] - latest_data['Open']) / latest_data['Open']) * 100
        }

    def stock_market_report(self, symbols=None):
        if symbols is None:
            symbols = self.user_watch_list
        stock_data_list = []
        for symbol in symbols:
            if symbol and symbol != 'None':
                stock_data = self.get_stock_info(symbol)
                if 'error' not in stock_data:
                    stock_data_list.append(stock_data)
        sorted_stocks = sorted(stock_data_list, key=lambda x: abs(x['percent_change']), reverse=True)
        significant_changes = [data for data in sorted_stocks if abs(data['percent_change']) > 1]  # Threshold for significant change
        if not significant_changes:
            return "Most stocks haven't seen much movement. Here are the ones that are seeing the most action:"
        report = ["Here are the stocks with the most action:"]
        for data in significant_changes:
            change_type = "gained" if data['percent_change'] > 0 else "lost"
            report_line = f"{data['symbol']} has {change_type} {abs(data['percent_change']):.1f}%\n...\n"
            report.append(report_line)
        return '\n'.join(report)

    def get_industry_avg_pe(self, symbol):
        industry_average_pe = 25  # Default placeholder value
        return industry_average_pe

    def calculate_pe_ratio(self, symbol):
        stock = yf.Ticker(symbol)
        pe_ratio = stock.info.get('trailingPE')

        if pe_ratio is None:  # If trailing P/E is not available, try forward P/E
            pe_ratio = stock.info.get('forwardPE')

        return pe_ratio

    def is_undervalued(self, symbol, pe_ratio):
        if pe_ratio is None:
            return False  # If PE ratio data is not available, return False

        industry_avg_pe = self.get_industry_avg_pe(symbol)
        return pe_ratio < industry_avg_pe
    
    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_yearly_change(self, hist, years):
        """Calculate the percentage change over a specified number of years."""
        if len(hist) > 0:
            year_start = hist.iloc[0]['Close']
            year_end = hist.iloc[-1]['Close']
            return ((year_end - year_start) / year_start) * 100
        return 0

    def calculate_period_change(self, hist, period='1M'):
        """Calculate the percentage change over a specified recent period."""
        if len(hist) > 0:
            period_hist = None
            if period == '1M':
                period_hist = hist.loc[hist.index >= (hist.index.max() - pd.DateOffset(months=1))]
            elif period == '3M':
                period_hist = hist.loc[hist.index >= (hist.index.max() - pd.DateOffset(months=3))]
            if period_hist is not None and not period_hist.empty:
                period_start = period_hist.iloc[0]['Close']
                period_end = period_hist.iloc[-1]['Close']
                return ((period_end - period_start) / period_start) * 100
        return 0

    def is_buy_signal(self, hist, rsi, symbol):
        year_change = self.calculate_yearly_change(hist, 1)
        recent_change = self.calculate_period_change(hist, '1M')
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        pe_ratio = self.calculate_pe_ratio(symbol)

        if pe_ratio is not None:
            undervalued = self.is_undervalued(symbol, pe_ratio)
        else:
            undervalued = False  # If PE ratio data is not available, consider it not undervalued

        if (year_change > 5 and recent_change > 2 and 
            hist['Close'].iloc[-1] > ma50 and rsi < 70 and undervalued):
            reasons = []
            if year_change > 5: reasons.append(f"Yearly growth: {year_change:.1f}%. ...")
            if recent_change > 2: reasons.append(f"Monthly growth {recent_change:.1f}%. ...")
            if hist['Close'].iloc[-1] > ma50: reasons.append("Above 50-day average. ...")
            if rsi < 70: reasons.append(f"RSI: {rsi:.1f}. ...")
            if undervalued: reasons.append(f"P/E ratio: {pe_ratio:.1f}. ...")
            return True, " and ".join(reasons)
        return False, ""

    def is_sell_signal(self, hist, rsi, symbol):
        year_change = self.calculate_yearly_change(hist, 1)
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        pe_ratio = self.calculate_pe_ratio(symbol)
        industry_avg_pe = self.get_industry_avg_pe(symbol)

        # Check if pe_ratio is not None before comparing
        overvalued = False
        if pe_ratio is not None:
            overvalued = pe_ratio > industry_avg_pe * 1.2  # Assuming overvalued if 20% higher than industry average

        if (year_change < 0 or hist['Close'].iloc[-1] < ma50 or rsi > 70 or overvalued):
            reasons = []
            if year_change < 0: reasons.append(f"Yearly loss {year_change:.1f}%. ...")
            if hist['Close'].iloc[-1] < ma50: reasons.append("Below 50-day average. ...")
            if rsi > 70: reasons.append(f"RSI: {rsi:.1f}. ...")
            if overvalued: reasons.append(f"P/E ratio: {pe_ratio:.1f}. ...")
            return True, " or ".join(reasons)
        return False, ""

    def find_stock_recommendations(self):
        buy_recommendations = []
        sell_recommendations = []
        hold_recommendations = []
        for symbol in self.user_watch_list:
            if symbol == 'None':
                continue
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if not hist.empty:
                rsi = self.calculate_rsi(hist['Close']).iloc[-1]
                # Pass the 'symbol' argument to is_buy_signal and is_sell_signal
                buy_signal, buy_reason = self.is_buy_signal(hist, rsi, symbol)
                sell_signal, sell_reason = self.is_sell_signal(hist, rsi, symbol)
                report_line = f"{symbol}: Recommendation: "
                if buy_signal:
                    buy_recommendations.append(report_line + f"Buy. WHY: ... {buy_reason}\n...\n")
                elif sell_signal:
                    sell_recommendations.append(report_line + f"Sell. WHY: ... {sell_reason}\n...\n")
                else:
                    hold_recommendations.append(report_line + "Hold\n...\n")
        categorized_recommendations = (
            ["\nBuy Recommendations:\n"] + buy_recommendations +
            ["\nSell Recommendations:\n"] + sell_recommendations +
            ["\nHold Recommendations:\n"] + hold_recommendations
        )
        return '\n'.join(categorized_recommendations) if any([buy_recommendations, sell_recommendations, hold_recommendations]) else "No recommendations available."

    def find_discounted_stocks(self):
        discounted_stocks_report = []
        for symbol in self.user_watch_list:
            if symbol == 'None':  # Skip if the symbol is 'None'
                continue
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if not hist.empty:
                year_start = hist.iloc[0]['Close']  # Yearly change calculation
                year_end = hist.iloc[-1]['Close']
                year_change = ((year_end - year_start) / year_start) * 100
                recent_high = hist['Close'].max()  # Discount from high calculation
                current_price = year_end
                discount_from_high = ((recent_high - current_price) / recent_high) * 100
                ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]  # Moving averages calculation
                ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                rsi = self.calculate_rsi(hist['Close']).iloc[-1]  # RSI calculation
                # Volume trend (optional)
                volume_increase = hist['Volume'].iloc[-1] > hist['Volume'].mean()  # Volume trend (optional)
                # Criteria check
                if (year_change > 5 and 3 <= discount_from_high <= 25 and
                    (current_price > ma50 or current_price > ma200) and rsi < 40):
                    report_line = f"{symbol}: Yearly Change: {year_change:.1f}%, Discount: {discount_from_high:.1f}%\n...\n"
                    discounted_stocks_report.append(report_line)
        return '\n'.join(discounted_stocks_report) if discounted_stocks_report else "No discounted stocks found meeting the criteria."

# FUNCTION DEFINITIONS ###################################################################################################################################

# Main speech_recognition function
def parse_user_speech():
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
        print('Listening timed out. Please try again.')
        return None

    except sr.UnknownValueError:
        print('Speech not recognized. Please try again.')
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
    print(f'{text}\n')

def control_mouse(action, direction=None, distance=0):
    if action == 'click':
        pyautogui.click()
    elif action == 'move':
        if direction == 'north':
            pyautogui.move(0, -distance, duration=0.1)
        elif direction == 'south':
            pyautogui.move(0, distance, duration=0.1)
        elif direction == 'west':
            pyautogui.move(-distance, 0, duration=0.1)
        elif direction == 'east':
            pyautogui.move(distance, 0, duration=0.1)
            
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

# Gathering a summary of a wikipedia page based on user input
def wiki_summary(query = ''):
    search_results = wikipedia.search(query)
    if not search_results:
        print('No results found.')
    try:
        wiki_page = wikipedia.page(search_results[0])
    except wikipedia.DisambiguationError as e:
        wiki_page = wikipedia.page(e.options[0])
    print(wiki_page.title)
    wiki_summary = str(wiki_page.summary)
    speak_mainframe(wiki_summary)

# Querying Wolfram|Alpha based on user input
def wolfram_alpha(query):
    wolfram_client = wolframalpha.Client(wolfram_app_id)
    try:
        response = wolfram_client.query(query)
        print(f"Response from Wolfram Alpha: {response}")

        # Check if the query was successfully interpreted
        if not response['@success']:
            suggestions = response.get('didyoumeans', {}).get('didyoumean', [])
            if suggestions:
                # Handle multiple suggestions
                if isinstance(suggestions, list):
                    suggestion_texts = [suggestion['#text'] for suggestion in suggestions]
                else:
                    suggestion_texts = [suggestions['#text']]

                suggestion_message = " or ".join(suggestion_texts)
                speak_mainframe(f"Sorry, I couldn't interpret that query. These are the alternate suggestions: {suggestion_message}.")
            else:
                speak_mainframe('Sorry, I couldn\'t interpret that query. Please try rephrasing it.')

            return 'Query failed.'

        relevant_pods_titles = [
            "Result", "Definition", "Overview", "Summary", "Basic information",
            "Notable facts", "Basic properties", "Notable properties",
            "Basic definitions", "Notable definitions", "Basic examples",
            "Notable examples", "Basic forms", "Notable forms",
            "Detailed Information", "Graphical Representations", "Historical Data",
            "Statistical Information", "Comparative Data", "Scientific Data",
            "Geographical Information", "Cultural Information", "Economic Data",
            "Mathematical Proofs and Derivations", "Physical Constants",
            "Measurement Conversions", "Prediction and Forecasting", "Interactive Pods"]

        # Filtering and summarizing relevant pods
        answer = []
        for pod in response.pods:
            if pod.title in relevant_pods_titles and hasattr(pod, 'text') and pod.text:
                answer.append(f"{pod.title}: {pod.text}")

        # Create a summarized response
        response_text = ' '.join(answer)
        if response_text:
            speak_mainframe(response_text)
        else:
            speak_mainframe("I found no information in the specified categories.")

        # Asking user for interest in other pods
        for pod in response.pods:
            if pod.title not in relevant_pods_titles:
                speak_mainframe(f"Do you want to hear more about {pod.title}? Say 'yes' or 'no'.")
                user_input = parse_user_speech().lower()
                if user_input == 'yes' and hasattr(pod, 'text') and pod.text:
                    speak_mainframe(pod.text)
                    continue
                elif user_input == 'no':
                    break

        return response_text

    except Exception as e:
        print(f"An error occurred: {e}")
        speak_mainframe('An error occurred while processing the query.')
        return f"An error occurred: {e}"

# Get a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location
def get_weather_forecast():
    appid = f'{open_weather_api_key}'

    # Fetching coordinates from environment variables
    lat = USER_SELECTED_HOME_LAT
    lon = USER_SELECTED_HOME_LON

    # OpenWeatherMap API endpoint for 4-day hourly forecast
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={appid}"

    response = requests.get(url)
    print("Response status:", response.status_code)
    if response.status_code != 200:
        return "Failed to retrieve weather data."

    data = response.json()
    print("Data received:", data)

    # Process forecast data
    forecast = ""
    timezone = pytz.timezone(USER_SELECTED_TIMEZONE)
    now = datetime.now(timezone)
    periods = [(now + timedelta(days=i)).replace(hour=h, minute=0, second=0, microsecond=0) for i in range(4) for h in [6, 12, 18, 0]]

    for i in range(0, len(periods), 4):
        day_forecasts = []
        for j in range(4):
            start, end = periods[i + j], periods[i + j + 1] if j < 3 else periods[i] + timedelta(days=1)
            period_forecast = [f for f in data['list'] if start <= datetime.fromtimestamp(f['dt'], tz=timezone) < end]
            
            if period_forecast:
                avg_temp_kelvin = sum(f['main']['temp'] for f in period_forecast) / len(period_forecast)
                avg_temp_fahrenheit = (avg_temp_kelvin - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
                descriptions = set(f['weather'][0]['description'] for f in period_forecast)
                time_label = ["morning", "afternoon", "evening", "night"][j]
                day_forecasts.append(f"{time_label}: average temperature {avg_temp_fahrenheit:.1f}°F, conditions: {', '.join(descriptions)}")

        if day_forecasts:
            forecast_date = periods[i].strftime('%Y-%m-%d')
            # Convert forecast_date to weekday format aka "Monday", etc.
            forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').strftime('%A')
            forecast += f"\n{forecast_date}: {'; '.join(day_forecasts)}."

    return forecast

def gemini_chat():
    speak_mainframe('OK.')
    time.sleep(1)
    chat = gemini_model.start_chat(history=[])
    
    prompt_template = '''# System Message # - Gemini, you are in a verbal chat with the user via a 
    STT / TTS application. Please generate your text in a way that sounds like natural speech 
    when it's spoken by the TTS app. Please avoid monologuing or including anything in the output that will 
    not sound like natural spoken language. After confirming you understand this message, the chat will proceed. Please 
    confirm your understanding of these instructions by simply saying "Chat loop is open."'''

    intro_response = chat.send_message(f'{prompt_template}', stream=True)
    if intro_response:
        speak_mainframe(f"Hi {USER_PREFERRED_NAME}. ")
        time.sleep(.25)
        for chunk in intro_response:
            speak_mainframe(chunk.text)
    time.sleep(1)
    
    while True:
        user_input = parse_user_speech()
        if not user_input:
            continue

        query = user_input.lower().split()
        if not query:
            continue

        if query[0] == activation_word and query[1] == 'terminate' and query[2] == 'chat':
            speak_mainframe('Ending Gemini chat.')
            break
        else:
            response = chat.send_message(f'{user_input}', stream=True)
            if response:
                for chunk in response:
                    speak_mainframe(chunk.text)
                    speak_mainframe(" ")
                    time.sleep(.1)
                time.sleep(1)
          
# MAIN LOOP ###################################################################################################################################

if __name__ == '__main__':
    stock_reports = StockReports(USER_STOCK_WATCH_LIST)
    threading.Thread(target=speech_manager, daemon=True).start()
    
    # # Prompt for password
    # speak_mainframe('Please say the password to continue.')
    # time.sleep(.5)
    # while True:
    #     spoken_password = parse_user_speech().lower()
    #     if not spoken_password:
    #         continue
    #     if spoken_password == password:
    #         speak_mainframe('Cool.')
    #         time.sleep(.5)
    #         break
    #     else:
    #         speak_mainframe('Try again.')
    
    speak_mainframe(f'{activation_word} online.')
    time.sleep(.5)
    
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
            time.sleep(.5)
            break

        # screenshot the screen
        if len(query) > 1 and query[0] == activation_word and query[1] == 'screenshot':
            today = datetime.today().strftime('%Y%m%d %H%M%S')         
            subprocess.call(['screencapture', 'screenshot.png'])
            # Save the screenshot to the file drop folder
            subprocess.call(['mv', 'screenshot.png', f'{FILE_DROP_DIR_PATH}/screenshot_{today}.png'])
            speak_mainframe('Saved.')
            time.sleep(.5)
            continue
            
        # Note taking
        if len(query) > 1 and query[0] == activation_word and query[1] == 'take' and query[2] == 'notes':
            today = datetime.today().strftime('%Y%m%d %H%M%S')         
            speak_mainframe('OK... What is the subject of the note?')
            time.sleep(.5)
            new_note_subject = parse_user_speech().lower().replace(' ', '_')
            speak_mainframe('OK, go ahead.')
            time.sleep(.5)
            new_note_text = parse_user_speech().lower()
            with open(f'{BASE_KNOWLEDGE_DIR_PATH}/notes_{new_note_subject}.txt', 'a') as f:
                f.write(f'{today}, {new_note_text}' + '\n')
            speak_mainframe('Saved.')
            time.sleep(.5)
            continue

        # recall notes
        if len(query) > 1 and query[0] == activation_word and query[1] == 'recall' and query[2] == 'notes':
            subject_pattern = r'notes_(\w+)\.txt'
            subjects = []
            for file in os.listdir(BASE_KNOWLEDGE_DIR_PATH):
                if file.startswith('notes_'):
                    subject = re.search(subject_pattern, file).group(1)
                    subjects.append(subject)
            speak_mainframe(f'These subjects are present in your notes: {subjects}. Please specify the subject of the note you want to recall.')
            time.sleep(1)
            desired_subject = parse_user_speech().lower().replace(' ', '_')
            with open(f'{BASE_KNOWLEDGE_DIR_PATH}/notes_{desired_subject}.txt', 'r') as f:
                note_contents = f.read()
                note = []
                for row in note_contents.split('\n'):  # Split each row into a list of words and begin speaking on the 3rd word of each row
                    row = row.split()
                    if len(row) >= 3:  # Check if there are at least 3 words in the row
                        note.append(' '.join(row[2:]))
                speak_mainframe(f'{note}')
            continue
        
        # google search
        if len(query) > 1 and query[0] == activation_word and query[1] == 'google' and query[2] == 'search':
            speak_mainframe('Heard')
            query = ' '.join(query[3:])
            url = f'https://www.google.com/search?q={query}'
            webbrowser.open(url, new=1)
            continue

        # Mouse control
        if len(query) > 2 and query[0] == activation_word and query[1] == 'mouse':
            if query[2] == 'click':
                control_mouse('click')
            elif query[2] in ['north', 'south', 'west', 'east'] and len(query) > 3 and query[3].isdigit():
                distance = int(query[3].replace(',', ''))
                control_mouse('move', query[2], distance)

        # translate
        if len(query) > 1 and query[0] == activation_word and query[1] == 'translate' and query[2] == 'to':
            target_language_name = query[3]
            speak_mainframe(f'Speak the phrase you want to translate.')
            time.sleep(1)
            phrase_to_translate = parse_user_speech().lower()
            translation, target_voice = translate(phrase_to_translate, target_language_name)
            speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
            continue

        # wikipedia summary
        if len(query) > 1 and query[0] == activation_word and query[1] == 'wiki' and query[2] == 'summary':
            speak_mainframe(f'What should we summarize from Wikipedia?')
            time.sleep(1)
            wikipedia_summary_query = parse_user_speech().lower()
            print("Wikipedia Query:", wikipedia_summary_query)  # Debugging print statement
            speak_mainframe(f'Searching {wikipedia_summary_query}')
            summary = wiki_summary(wikipedia_summary_query)  # Get the summary
            speak_mainframe(summary)  # Speak the summary
            continue

        # youtube video
        if len(query) > 1 and query[0] == activation_word and query[1] == 'youtube' and query[2] == 'video':
            speak_mainframe(f'What would you like to search for on YouTube?')
            youtube_query = parse_user_speech().lower()
            print("YouTube Query:", youtube_query)  # Debugging print statement
            speak_mainframe(f'Searching YouTube for {youtube_query}')
            url = f'https://www.youtube.com/results?search_query={youtube_query}'
            webbrowser.open(url)
            continue
        
        # wolfram alpha
        if len(query) > 1 and query[0] == activation_word and query[1] == 'wolfram' and query[2] == 'alpha':
            speak_mainframe(f'What would you like to ask?')
            wolfram_alpha_query = parse_user_speech().lower()
            speak_mainframe(f'Weird question but ok...')
            result = wolfram_alpha(wolfram_alpha_query)
            print(f'User: {wolfram_alpha_query} \nWolfram|Alpha: {result}')
            continue
                
        # open weather forecast
        if len(query) > 1 and query[0] == activation_word and query[1] == 'get' and query[2] == 'weather' and query[3] == 'forecast':
            speak_mainframe(f'OK - beginning weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}')
            weather_forecast = get_weather_forecast()
            print("Weather forecast:", weather_forecast)
            speak_mainframe(f'Weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}: {weather_forecast}')
            continue

        # stock market reports
        if len(query) > 1 and query[0] == activation_word and query[1] == 'stock' and query[2] == 'report':
            today = datetime.today().strftime('%Y%m%d %H%M%S')         
            speak_mainframe('Discounts, recommendations, yesterday, or single?')
            answer = parse_user_speech()
            if not answer:
                continue
            focus = answer.lower().split()
            if 'discounts' in focus:
                discounts_update = stock_reports.find_discounted_stocks()
                speak_mainframe(discounts_update)
                print(discounts_update)
                continue
            if 'recommendations' in focus:
                recs_update = stock_reports.find_stock_recommendations()
                speak_mainframe(recs_update)
                print(recs_update)
                continue
            if 'yesterday' in focus:
                portfolio_update = stock_reports.stock_market_report()
                speak_mainframe(portfolio_update)
                print(portfolio_update)
                continue
            if 'single' in focus:
                speak_mainframe('Please spell the stock ticker.')
                time.sleep(.5)
                attempts = 0
                while attempts < 3:
                    raw_stock_input = parse_user_speech().upper().strip()
                    stock = ''.join(raw_stock_input.split())
                    speak_mainframe(f"Did you say {stock}?")
                    confirmation = parse_user_speech().lower()
                    if 'yes' in confirmation:
                        stock_reports.handle_single_stock_query(stock)
                        break
                    else:
                        speak_mainframe("I didn't catch that. Could you please spell out the stock ticker again?")
                        attempts += 1
                if attempts == 3:
                    speak_mainframe("Having trouble recognizing the stock ticker. Let's try something else.")

        if len(query) > 2 and query[0] == activation_word and query[1] == 'call' and query[2] == 'gemini':
            gemini_chat()  
            
        # In your main loop
        if len(query) > 2 and query[0] == activation_word and query[1] == 'custom' and query[2] == 'search':
            speak_mainframe('Starting search assistant.')
            CustomSearchEngines.search_chat_bot()
            speak_mainframe('Search assistant deactivated.')
    
# END ###################################################################################################################################


