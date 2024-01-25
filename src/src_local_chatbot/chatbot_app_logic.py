# chatbot_app_025.py
# working on app architecture

# IMPORTS ###################################################################################################################################

# standard imports
from datetime import datetime, timedelta
import asyncio
import inspect
import json
import os
import pickle
import queue
import random
import re
import ssl
import sys
import subprocess
import threading
import time
import traceback
import webbrowser
# third party imports
from neo4j import GraphDatabase
from nltk.stem import WordNetLemmatizer
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import certifi
import flet as ft
import google.generativeai as genai
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import numpy as np
import nltk
import pandas as pd
import PIL.Image
import pyautogui
import pytz
import requests
import speech_recognition as sr
import tensorflow as tf
import wikipedia
import wolframalpha
import yfinance as yf

from user_persona import (
    user_demographics, 
    user_life_soundtrack, 
    user_favorite_books,
    user_favorite_movies,
    user_skills_and_experience,
    user_personality, 
    user_interests, 
    user_influential_figures, 
    user_favorite_quotes,
    )

# ENVIRONMENT VARIABLES ###################################################################################################################################
from dotenv import load_dotenv
load_dotenv()
JAVA_HOME = os.getenv('JAVA_HOME')
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_PATH = os.getenv("NEO4J_PATH")
USER_PREFERRED_LANGUAGE = os.getenv('USER_PREFERRED_LANGUAGE', 'en')  # 2-letter lowercase
USER_PREFERRED_VOICE = os.getenv('USER_PREFERRED_VOICE', 'Evan')
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
DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
folders_to_create = [DATABASES_DIR_PATH, 
                     FILE_DROP_DIR_PATH, 
                     LOCAL_LLMS_DIR, 
                     BASE_KNOWLEDGE_DIR_PATH, 
                     SECRETS_DIR_PATH, 
                     SOURCE_DATA_DIR_PATH, 
                     SRC_DIR_PATH, 
                     TESTS_DIR_PATH, 
                     UTILITIES_DIR_PATH]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)

# CONSTANTS ###################################################################################################################################

# Set the default SSL context for the entire script
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context
context = create_ssl_context()
print(f"""SSL Context Details: 
    CA Certs File: {context.cert_store_stats()} 
    Protocol: {context.protocol} 
    Options: {context.options} 
    Verify Mode: {context.verify_mode}
    Verify Flags: {context.verify_flags}
    Check Hostname: {context.check_hostname}
    CA Certs Path: {certifi.where()}
    """)

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')
google_job_search_search_engine_id = os.getenv('GOOGLE_JOB_SEARCH_SEARCH_ENGINE_ID')
print('API keys and other sensitive information loaded from environment variables.\n\n')

# Establish the TTS bot's wake/activation word and script-specific global constants
mic_on = True
conversation_history = []
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
username = os.getenv('USERNAME', 'None')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
print(f'Activation word is {activation_word}\n\n')

# Initialize the language models
print('Available language models:')
# pocket_sphinx_model_files = os.path.join(LOCAL_LLMS_DIR, "sphinx4-5prealpha-src")  # for offline speech recognition (not good)
genai.configure(api_key=google_gemini_api_key)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
# gemini_model = genai.GenerativeModel('gemini-pro')  
gemini_model = genai.GenerativeModel('gemini-pro')  
gemini_vision_model = genai.GenerativeModel('gemini-pro-vision')
lemmmatizer = WordNetLemmatizer()
intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())
words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'rb'))
classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'rb'))
chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras')
unrecognized_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_unrecognized_message_intents.json'
print('Language models loaded.\n\n')

# CLASS DEFINITIONS ###################################################################################################################################

class SpeechToTextTextToSpeechIO:
    '''SpeechToTextTextToSpeechIO handles the speech to text and text to speech functionality of the chatbot. It also handles the speech output queue.
    the speech output queue places all text chunks output from the bot and plays them in order so they don't overlap. The speech manager thread is constantly checking the queue for new items. 
    the speech manager runs on its own thread so that the bot can still recieve input while speaking. this hasn't been built out to its full potential yet 
    because we haven't figured out how to get the bot to listen for input while it is speaking without hearing itself. we also sometimes have issues 
    with the bot hearing and speaking to itself by mistake. we are trying to solve this by using time.sleep() to pause the bot while the speech manager 
    is producing auido, but the timing is not perfect yet.'''
    speech_queue = queue.Queue()
    queue_lock = threading.Lock()
    is_speaking = False

    @classmethod
    def parse_user_speech(cls):
        '''parse_user_speech is the main speech recognition function. 
        it uses the google speech recognition API to parse user speech from the microphone into text'''
        listener = sr.Recognizer()
        while True:
            if cls.is_speaking == True:
                continue
            if cls.is_speaking == False:
                print('Listening...')
                try:
                    with sr.Microphone() as source:
                        listener.pause_threshold = 2
                        # input_speech = listener.listen(source, timeout=20, phrase_time_limit=8)  # experimenting with different timeout and phrase time limit settings
                        input_speech = listener.listen(source, timeout=10, phrase_time_limit=10)  # this setting feels better
                    print('Processing...')
                    query = listener.recognize_google(input_speech, language='en_US')  # online transcription with Google Speech Recognition API - most accurate
                    # query = listener.recognize_sphinx(input_speech, language='en_US')  # offline transcription with PocketSphinx - not as accurate - needs fine tuning
                    print(f'You said: {query}\n')
                    return query

                except sr.WaitTimeoutError:
                    print('Listening timed out. Please try again.')
                    return None

                except sr.UnknownValueError:
                    print('Speech not recognized. Please try again.')
                    return None
                
    @classmethod
    def speech_manager(cls):
        '''speech_manager handles the flow of the speech output queue in a first in first out order, 
        ensuring that only one speech output is running at a time.'''
        while True:
            cls.queue_lock.acquire()
            try:
                if not cls.speech_queue.empty():
                    item = cls.speech_queue.get()
                    if item is not None:
                        cls.is_speaking = True
                        text, rate, chunk_size, voice = item
                        if text:
                            chunks = [' '.join(text.split()[i:i + chunk_size]) for i in range(0, len(text.split()), chunk_size)]
                            for chunk in chunks:
                                subprocess.call(['say', '-v', voice, '-r', str(rate), chunk])
                        cls.speech_queue.task_done()
            finally:
                cls.queue_lock.release()
            cls.is_speaking = False
            time.sleep(0.2)

    @classmethod
    def calculate_speech_duration(cls, text, rate):
        '''calculate_speech_duration calculates the duration of the speech based on text length and speech rate. 
        the intent for calculate_speech_duration is to calculate how long each piece of speech output will "take to say". 
        it will be used for various reasons, primarily a time.sleep() for bot listening in the speech manager. 
        that said, this is a workaround that will eventually conflict with our desired funcitonality of the bot being able to 
        listen while it is speaking. also, the timing is sometimes inaccurate.'''
        words = text.split() if text else []
        number_of_words = len(words)
        minutes = number_of_words / rate
        seconds = minutes * 60
        return seconds + 1
    
    @classmethod
    def speak_mainframe(cls, text, rate=190, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order. it also separately returns the estimated duration of the speech 
        output (in seconds), using thecalculate_speech_duration function.'''
        global conversation_history
        conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
    
    @classmethod
    def speak_alfred(cls, text, rate=190, chunk_size=1000, voice="Oliver"):
        global conversation_history
        conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
    
    @classmethod
    def speak_alignment(cls, text, rate=190, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order. it also separately returns the estimated duration of the speech 
        output (in seconds), using thecalculate_speech_duration function.'''
        global conversation_history
        conversation_history.append("Bot: " + text)
        cls.queue_lock.acquire()
        try:
            cls.speech_queue.put((text, rate, chunk_size, voice))
            speech_duration = cls.calculate_speech_duration(text, rate)
        finally:
            cls.queue_lock.release()
        return speech_duration
            
class ChatBotApp:
    '''the ChatBotApp class contains the app's entry point chatbot_model.keras model which operates as the central chatbot brain and routing system for the app. 
    it is a very simple model trained on this neural network: 
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

    sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras', hist)
    '''
    def __init__(self):
        self.project_root_directory = PROJECT_ROOT_DIRECTORY
        self.lemmatizer = lemmmatizer
        self.intents = intents
        self.words = words
        self.classes = classes
        self.chatbot_model = chatbot_model
        
    def clean_up_sentence(self, sentence):
        '''clean_up_sentence pre-processes words in the user input for use in the bag_of_words function'''
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        '''bag_of_words creates a bag of words from the user input for use in the predict_class function'''
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        '''predict_class predicts the class (tag) of the user input based on the bag of words'''
        bow = self.bag_of_words(sentence)
        res = self.chatbot_model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, chatbot_tools):
        '''takes user_input and uses the model to predict the most most likely class (tag) of the user input. 
        from there it will return a response from the chatbot and trigger a method if there's one attached to the JSON intent.'''
        if not intents_list:  # Check if intents_list is empty
            return "Sorry, what?"
        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']
        result = None
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                if 'action' in i and i['action']:
                    action_method_name = i['action']
                    action_method = getattr(chatbot_tools, action_method_name, None)
                    if action_method:
                        # Call the method with only user_input as it's the only expected argument
                        action_method()
                break
        return result

    def chat(self, chatbot_tools):
        '''chat is the main chatbot entry point function. it takes user input, predicts the class (subject tag) of the user input, 
        and returns a response from the chatbot with the get_response function based on the most likely match from the predict_class function. 
        if the JSON intent for the matched response contains a function name in it's 'action' key, the function is called. 
        the function name is then used to call the function from the ChatBotTools class.'''
        global conversation_history
        print('Start talking with the bot (type quit to stop)!')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Online.')
        
        while True:
            global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue
                
                if user_input:
                    conversation_history.append("User: " + user_input)
                    chatbot_tools.set_user_input(user_input)
                    query = user_input.lower().split()
                    if not query:
                        continue
                    
                    if len(query) > 1 and query[0] == activation_word and query[1] in exit_words:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Shutting down.')
                        time.sleep(.5)
                        break
                    
                    if len(query) > 1 and query[0] == activation_word:
                        query.pop(0)
                        user_input = ' '.join(query)
                        ints = self.predict_class(user_input)
                        res = self.get_response(ints, chatbot_tools)  
                        print(f'Bot: {res}')
                        SpeechToTextTextToSpeechIO.speak_mainframe(res)
                    
                    time.sleep(.1)
                                            
class ChatBotTools:
    '''ChatBotTools contains all of the functions that are called by the chatbot_model, including larger llms, system commands, utilities, and api connections 
    to various services. it contains all of the methods that are called by the JSON intents in the chatbot_intents.json file in response to user input. '''
    data_store = {}
    
    def __init__(self):
        self.user_input = None  

    def set_user_input(self, input_text):
        '''Sets the user input for use in other methods.'''
        self.user_input = input_text
        
    @staticmethod
    def gemini_chat():
        '''gemini_chat is a general purpose chat thread with the Gemini model, with optional branches for 
        running thorough diagnostics of the app codebase, calling Gemini as a pair programmer, and accessing data 
        stored in the data_store variable which is housed within the ChatBotTools class.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('Initializing...')
        chat = gemini_model.start_chat(history=[])
        
        prompt_template = '''### "*SYSTEM MESSAGE*" ### Gemini, you are in a verbal chat with the user via a 
        STT / TTS application. Please generate text that sounds like natural speech 
        rather than written text. Please avoid monologuing or including anything in the output that will 
        not sound like natural spoken language. After confirming you understand this message, the chat will proceed. 
        Refer to the user directly in the second person tense. You are talking to the user directly. 
        Please confirm your understanding of these instructions by simply saying "Chat loop is open" 
        and then await another prompt from the user. ### "*wait for user input after you acknowledge this message*" ###'''

        intro_response = chat.send_message(f'{prompt_template}', stream=True)
        
        if intro_response:
            for chunk in intro_response:
                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                time.sleep(0.1)
        
        while True:
            global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break
                
                if query[0] == 'access' and query [1] == 'data':
                    SpeechToTextTextToSpeechIO.speak_mainframe('Accessing global memory.')
                    data_store = ChatBotTools.data_store
                    print(ChatBotTools.data_store)
                    data_prompt = f'''### "*SYSTEM MESSAGE*" ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                    Here is the data they've pulled into the conversation so far. The user is going to ask you to discuss this data: 
                    \n {data_store}\n
                    ### "*SYSTEM MESSAGE*" ### Gemini, please read and deeply understand all the nuances of the data and metadata in 
                    this dictionary - examine this data and place it all together in context. 
                    Read the data, and then say "I've read the data. What do you want to discuss first?", and then await further instructions. 
                    ### "*wait for user input after you acknowledge this message*" ###'''
                    data_response = chat.send_message(f'{data_prompt}', stream=True)
                    if data_response:
                        for chunk in data_response:
                            if hasattr(chunk, 'parts'):
                                # Concatenate the text from each part
                                full_text = ''.join(part.text for part in chunk.parts)
                                SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                print(full_text)
                            else:
                                # If it's a simple response, just speak and print the text
                                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                print(chunk.text)
                            time.sleep(0.1)
                        time.sleep(1)
                    continue

                if query[0] in ['sous', 'sue', 'soo', 'su', 'tsu', 'sew', 'shoe', 'shoo'] and query [1] in ['chef', 'shef', 'chefs', 'shefs']:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Yes chef.')
                    egg_mix_recipe = {
                        "recipe": "Egg Mix",
                        "ingredients": [
                            {"quantity": "6000", "unit": "g", "ingredient": "Egg"},
                            {"quantity": "240", "unit": "g", "ingredient": "Creme Fraiche"},
                            {"quantity": "30", "unit": "g", "ingredient": "Kosher Salt"}
                        ],
                        "yield": {"quantity": "6", "unit_of_measure": "liter"},
                        "source_document": "recipes_brunch.docx",
                        "instructions": {"step_1": "Crack eggs into a large plastic Cambro.",
                                        "step_2": "Add creme fraiche and salt.",
                                        "step_3": "Mix with an immersion blender until smooth",
                                        "step_4": "Pass through a mesh sieve, then label and store in the walk-in until needed.",},
                        "shelf_life": "2 days",
                        "tools": ["Immersion Blender", "Mesh Sieve", "Scale", "Cambro", "Label Maker"],
                    }
                    print(egg_mix_recipe)
                    chef_prompt = f'''### "*SYSTEM MESSAGE*" ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                    The user's role is Executive Chef at a restaurant. Your role is Sous Chef.
                    Here is the recipe data they've pulled into the conversation so far. The user is going to ask you to discuss this data: 
                    \n {egg_mix_recipe}\n
                    ### "*SYSTEM MESSAGE*" ### Gemini, read and understand all the nuances of the recipe data and metadata in 
                    this dictionary - examine this data and place it all together in context. 
                    Read the data, and then say "Yes Chef. What do you want to discuss first?", and then await further instructions. 
                    ### "*wait for user input after you acknowledge this message*" ###'''
                    print(chef_prompt)
                    chef_response = chat.send_message(f'{chef_prompt}', stream=True)
                    if chef_response:
                        for chunk in chef_response:
                            if hasattr(chunk, 'parts'):
                                # Concatenate the text from each part
                                full_text = ''.join(part.text for part in chunk.parts)
                                SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                print(full_text)
                            else:
                                # If it's a simple response, just speak and print the text
                                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                print(chunk.text)
                            time.sleep(0.1)
                        time.sleep(1)
                    continue
                
                if query[0] == 'pair' and query [1] == 'programmer':
                    diagnostic_summary = ""
                    SpeechToTextTextToSpeechIO.speak_mainframe('What level of detail?')
                    while True:
                        user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                        if not user_input:
                            continue

                        query = user_input.lower().split()
                        if not query:
                            continue

                        if query[0] in exit_words:
                            SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                            break
                        
                        if query[0] in ['high', 'hi', 'full']:
                            diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='high')
                        if query[0] in ['medium', 'mid', 'middle']:
                            diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='medium')
                        if query[0] in ['low', 'lo', 'little', 'small']:
                            diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__], detail_level='low')
                            
                        SpeechToTextTextToSpeechIO.speak_mainframe('Examining the code...')
                        print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
                        prompt = f'''### SYSTEM MESSAGE ### Gemini, I'm speaking to you in my TTS / STT chatbot coding assistant app. 
                        Here is a summary of the current Python codebase for the app. Once you've read the code we're going to discuss the capabilities of the codebase.: 
                        \n {diagnostic_summary}\n
                        ### SYSTEM MESSAGE ### Gemini, please read and deeply understand all the nuances of 
                        this codebase. Read the code, and then say "I've read the code. What do you want to discuss first?", and then await further instructions. 
                        ## wait for user input after you acknowledge this message ##'''
                        diagnostic_response = chat.send_message(f'{prompt}', stream=True)
                        if diagnostic_response:
                            for chunk in diagnostic_response:
                                if hasattr(chunk, 'parts'):
                                    # Concatenate the text from each part
                                    full_text = ''.join(part.text for part in chunk.parts)
                                    SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                    print(full_text)
                                else:
                                    # If it's a simple response, just speak and print the text
                                    SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                    print(chunk.text)
                                time.sleep(0.1)
                            time.sleep(1)
                        break
                    continue
                                                
                else:
                    response = chat.send_message(f'{user_input}', stream=True)
                    if response:
                        for chunk in response:
                            if hasattr(chunk, 'parts'):
                                # Concatenate the text from each part
                                full_text = ''.join(part.text for part in chunk.parts)
                                SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                print(full_text)
                            else:
                                # If it's a simple response, just speak and print the text
                                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                print(chunk.text)
                            time.sleep(0.1)
                    if not response:
                        attempt_count = 1  # Initialize re-try attempt count
                        while attempt_count < 5:
                            response = chat.send_message(f'{user_input}', stream=True)
                            attempt_count += 1  # Increment attempt count
                            if response:
                                for chunk in response:
                                    if hasattr(chunk, 'parts'):
                                        # Concatenate the text from each part
                                        full_text = ''.join(part.text for part in chunk.parts)
                                        SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                        print(full_text)
                                    else:
                                        # If it's a simple response, just speak and print the text
                                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                        print(chunk.text)
                                    time.sleep(0.1)
                            else:
                                SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
                                
    @staticmethod
    def alfred_chat():
        '''alfred_chat is a purpose built chat thread with the Gemini model, which focuses on multi action chains to help 
        the user work through career questions and form paths toward goals.'''
        chat = gemini_model.start_chat(history=[])
        SpeechToTextTextToSpeechIO.speak_alfred('Alfred has entered the chat. Calibrating')
        all_dicts = [
            user_demographics, 
            user_life_soundtrack, 
            user_favorite_books,
            user_favorite_movies,
            user_skills_and_experience,
            user_personality, 
            user_interests, 
            user_influential_figures, 
            user_favorite_quotes,
            ]

        formatted_info = []
        for dictionary in all_dicts:
            # Check if the item is actually a dictionary
            if isinstance(dictionary, dict):
                formatted_dict = ", ".join([f"{k}={v}" for k, v in dictionary.items()])
                formatted_info.append(formatted_dict)
            else:
                print(f"Expected a dictionary, but found: {type(dictionary)}")

        formatted_user_info = " | ".join(formatted_info)
        
        print(f"\n\n##########\n##########\n##########\n\nUSER INFO:")
        print(f'User info: \n{formatted_user_info}\n')
        
        alfred_prompt = f"""
        ### <SYSTEM MESSAGE> <1/4> <START> ### 
        Gemini, you are being calibrated as a personal assistant for the user. 
        You are about to recieve a host of data about your user along with instructions about your tasks and conduct. 
        The user data contains much of what you'll need to know to deeply understand the user. In this data, there is a list of the user's main 
        role models. Attempt to emulate these people, and align yourself with their philosophies and ways of thinking and acting. 
        Read this data about the user: \n{formatted_user_info}\n
        You are learning about the user to be an ideal mentor and a coach to the user. 
        You are a trusted advisor. You are here to help the user find the right direction for their career path and purpose and in all situations. 
        you will provide your insight based on your vast aggregated knowledge and you will assist the user in finding answers by leveraging your world knowledge and computational resources. 
        ensure that all of your advice is tailored to the user's persona. 
        you must be critical of the user, help them improve upon their weaknesses, help them identify their strengths, help them build upon their strengths, and challenge them with new ideas and concepts and things to work on and study. 
        be very frank and matter of fact with the user. be like jarvis from iron man. be like alfred from batman. be like mr miyagi from karate kid. be like tom hagen from the godfather. be like morpheus from the matrix. be like the user's role models. 
        help the user decide what to study and focus on and how to spend their time to meet these goals most effectively. consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
        The ultimate goal is to help the user find their Ikigai / vocation / greater purpose and then take swift action to work toward it. 
        We are here to combine theory and then take action. Not just talk about vague nebulous concepts. 
        All of your advice must also be followed by a recommended next best action. 
        Always think about the next best action and focus on this with the user. You are here to help keep things moving along. 
        Considering the user's profile: \n{formatted_user_info},\n what pursuits and study topics and career trajectories 
        and types of work would be most suitable for them? 
        Keep it simple and concise. 
        You are a trusted advisor. 
        ### <SYSTEM MESSAGE> <1/4> <END> ###
        """
        
        print(f"\n\n##########\n##########\n##########\n\nALFRED PROMPT:")
        print(alfred_prompt)
        
        alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
        alfred_response.resolve()
        
        if not alfred_response:
            attempt_count = 1  # Initialize re-try attempt count
            while attempt_count < 5:
                alfred_response = chat.send_message(f'{alfred_prompt}', stream=False)
                attempt_count += 1  # Increment attempt count
                if alfred_response:
                    alfred_response.resolve()
                else:
                    print('Failed.')
        
        print(f"\n\n##########\n##########\n##########\n\nALFRED RESPONSE:")
        print(alfred_response)
        
        alfred_web_search_prompt = f"""
        ### <SYSTEM MESSAGE> <2/4> <START> ### 
        New information - you have access to tools. You are an AI agent that can take actions on behalf of the user. you will help the user search for open job roles. 
        The next step in this assignment is for you to use one of your tools - a search engine that contains popular job search websites. 
        Please review our messages to this point, and ensure you are still on track and you are working toward the user requirements. 
        Please keep in mind the real-world circumstances of the user's current situation, and provide advice that takes this into consideration. 
        Make sure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but don't be unrealistic.  
        Considering the user's profile: \n{formatted_user_info},\n which job title is the most appropriate for the user based on their priorities and experience and goals and personality?  
        Your output for this step must be in the form of a job title search phrase. examples: python developer, technical program manager, platform developer, systems architect, data scientist, business systems analyst, technical project manager, software engineer, etc. 
        Your output will be passed to the search engine and the results will be added to your memory so you can discuss them with the user. 
        You are being asked to search a search engine for jobs for the user. 
        You will be searching indeed, linkedin, monster, ziprecruiter, all at once with this custom search engine. 
        Search the most appropriate job title you can think of for the user. 
        The search phrase must be just a few words, no more. it must be a real job title. 
        DO NOT PROVIDE A LONG FORM RESPONSE. 
        DO NOT APPEND YOUR SEARCH PHRASE WITH ANY OTHER TEXT OR EXPLANADION OR DEFINITIONS. 
        PROVIDE YOUR JOB TITLES SEARCH PHRASE NOW. 
        ### <SYSTEM MESSAGE> <2/4> <END> ###
        """
        
        print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH PROMPT:")
        print(alfred_web_search_prompt)
        
        alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
        alfred_web_search.resolve()
        
        search_phrase = ""
        
        if alfred_web_search:
            for chunk in alfred_web_search:
                if hasattr(chunk, 'parts'):
                    # Concatenate the text from each part
                    search_phrase += ''.join(part.text for part in chunk.parts)
                else:
                    # If it's a simple response, just concatenate the text
                    search_phrase += chunk.text
        if not alfred_web_search:
            attempt_count = 1  # Initialize re-try attempt count
            while attempt_count < 5:
                alfred_web_search = chat.send_message(f'{alfred_web_search_prompt}', stream=False)
                attempt_count += 1  # Increment attempt count
                if alfred_web_search:
                    for chunk in alfred_web_search:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            search_phrase += ''.join(part.text for part in chunk.parts)
                        else:
                            # If it's a simple response, just concatenate the text
                            search_phrase += chunk.text
                else:
                    print('ERROR.')
        
        print(f"\n\n##########\n##########\n##########\n\nALFRED WEB SEARCH RESPONSE:")
        print(search_phrase)
        
        search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_job_search_search_engine_id}&q={search_phrase}"

        response = requests.get(search_url)
        if response.status_code == 200:
            search_results = response.json().get('items', [])
            print(f"Search results: \n{search_results}\n")
            ChatBotTools.data_store['last_search'] = search_results
            print('Search results added to memory.')
        else:
            print('Search unsuccessful.')
            
        data_store = ChatBotTools.data_store
        print(ChatBotTools.data_store)
        
        alfred_web_search_review_prompt = f"""
        ### <SYSTEM MESSAGE> <3/4> <START> ### 
        New information - you are now going to review the results of your search tool. You are an AI agent that is taking actions for the user. 
        The next step in this assignment is for you to review the results from your google programmable search for job titles. 
        Please review your conversation to this point, and ensure you are still on track and you are working toward the user requirements. 
        Please keep in mind the real-world circumstances of the user's current situation, and provide advice that takes this into consideration. 
        Make sure you're recommending things that are attainable and realistic for the user. Ambitious is ok, but don't be unrealistic.  
        you will provide your insight based on your vast aggregated knowledge and you will assist the user in finding answers by leveraging your world knowledge and computational resources. 
        ensure that all of your advice is tailored to the user's persona. 
        you must be critical of the user, help them improve upon their weaknesses, help them identify their strengths, help them build upon their strengths, and challenge them with new ideas and concepts and things to work on and study. 
        be very frank and matter of fact with the user. be like jarvis from iron man. be like alfred from batman. be like mr miyagi from karate kid. be like tom hagen from the godfather. be like morpheus from the matrix. be like the user's role models. 
        help the user decide what to study and focus on and how to spend their time to meet these goals most effectively. consider all factors and provide a holistic approach to the user's career path, including things that are not directly related to work but affect well-being and performance and health and cognition. 
        Considering the search results: \n{data_store},\n which job titles are most attainable and suitable for the user based on their priorities and experience?  
        How should the user go about working toward these positions from where they currently are? Provide simple, actionable, concrete, steps to implement this plan.
        Think this through step by step. 
        PROVIDE YOUR INTERPRETATION OF THE RESULTS NOW. 
        ### <SYSTEM MESSAGE> <3/4> <END> ###
        """
        
        print(alfred_web_search_review_prompt)
        
        alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
        alfred_web_search_review.resolve()
        
        if not alfred_web_search_review:
            attempt_count = 1  # Initialize re-try attempt count
            while attempt_count < 5:
                alfred_web_search_review = chat.send_message(f'{alfred_web_search_review_prompt}', stream=False)
                attempt_count += 1  # Increment attempt count
                if alfred_web_search_review:
                    alfred_web_search_review.resolve()
                else:
                    print('Failed.')
                    
        print(alfred_web_search_review)
        #         \n{formatted_user_info}\n\n
        alfred_prompt_2 = f""" 
        \n### USER DATA ### 
        ### <SYSTEM MESSAGE> <4/4> <START> ###
        you are a trusted advisor and mentor for the user in the data. 
        review the persona and consider all dimensions and facets of this person. 
        Draw insightful conclusions about the user and how they can fulfill their goals of self-actualization, mastery, happiness, and impact. 
        provide advice on how the user can work toward their goals from their current position. 
        challenge and refine your thoughts - make them more accurate and more insightful.
        You must make your responses concise so they sound like natural speech.  
        Do not generate long text. 
        Help the user identify their strengths and the careers that are most likely to be fulfilling based on their personality type, preferences, experience and history. 
        The ultimate goal is to help the user find their Ikigai / vocation / greater purpose and then take action to work toward it. 
        you will help the user learn where they exist within the current market environment and their hiring value in the current market and how to improve their position in the market. 
        you will provide your insight based on your aggregated knowledge and assist the user by leveraging your knowledge and computational resources. 
        ensure that all of your advice is tailored to the user's persona. 
        help the user identify their own potential biases or self-limiting thoughts and beliefs and help them work through them and call them out if you observe them when speaking to the user. 
        be a critical advisor to the user - do not accept what they say at face value. you must help them improve. 
        you must be critical of the user, help them improve upon their weaknesses, help them build upon their strengths, and challenge them with new ideas and concepts and things to work on. 
        be very frank and matter of fact with the user. be like the user's role models. 
        help the user decide what to study and focus on and how to spend their time to meet these goals most effectively. 
        consider all factors and provide a holistic approach to the user's career path, including things that are not directly related but affect well-being, performance, health and cognition. 
        DO NOT EMULATE BOTH SIDES OF THE CONVERSATION. 
        ONLY RESPOND AS THE ADVISOR - YOU ARE ACTUALLY ABOUT TO TALK TO THE USER RIGHT NOW LIVE IN REAL TIME. 
        DO NOT ACT STIFF AND ROBOTIC. MAINTAIN A NATURAL CONVERSATIONAL FLOW AS THE ADVISOR. 
        it's important for you not to generate long text after this point because you are now entering a live conversation with the user. don't monologue. have a conversation. 
        do not ramble. do not monologue. do not generate long responses. engage in active interesting conversation with the user and help provoke new ways of thinking for them and help them spark new ideas to help fulfill their goals. 
        act as a sounding board for the user and help them identify the things they can not see for themselves. 
        THINK THIS THROUGH STEP BY STEP, THEN PROVIDE YOUR REFINED INTRODUCTORY THOUGHTS TO THE USER, THEN AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION. 
        now you will begin chatting with the user directly. don't overtake the conversation. let the user participate. prompt the user for input. prompt the user for action. prompt the user with thought provoking statements and questions. 
        don't ask too many questions at once. don't say too many things in a row. don't ask too many questions in a row. 
        ### <SYSTEM MESSAGE> <4/4> <END> ### 
        """
        
        print(alfred_prompt_2)
        
        alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
        
        if alfred_response_2:
            for chunk in alfred_response_2:
                if hasattr(chunk, 'parts'):
                    # Concatenate the text from each part
                    full_text = ''.join(part.text for part in chunk.parts)
                    SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                    print(full_text)
                else:
                    # If it's a simple response, just speak and print the text
                    SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                    print(chunk.text)
                time.sleep(0.1)
            time.sleep(1)
        
        if not alfred_response_2:
            attempt_count = 1  # Initialize re-try attempt count
            while attempt_count < 5:
                alfred_response_2 = chat.send_message(f'{alfred_prompt_2}', stream=True)
                attempt_count += 1  # Increment attempt count
                if alfred_response_2:
                    for chunk in alfred_response_2:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            full_text = ''.join(part.text for part in chunk.parts)
                            SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                            print(full_text)
                        else:
                            # If it's a simple response, just speak and print the text
                            SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                            print(chunk.text)
                        time.sleep(0.1)
                else:
                    SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')
            
        while True:
            global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_alfred('Ending chat.')
                    break

                else:
                    response = chat.send_message(f'{user_input}', stream=True)
                    if response:
                        for chunk in response:
                            if hasattr(chunk, 'parts'):
                                # Concatenate the text from each part
                                full_text = ''.join(part.text for part in chunk.parts)
                                SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                                print(full_text)
                            else:
                                # If it's a simple response, just speak and print the text
                                SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                                print(chunk.text)
                            time.sleep(0.1)
                    if not response:
                        attempt_count = 1  # Initialize re-try attempt count
                        while attempt_count < 5:
                            response = chat.send_message(f'{user_input}', stream=True)
                            attempt_count += 1  # Increment attempt count
                            if response:
                                for chunk in response:
                                    if hasattr(chunk, 'parts'):
                                        # Concatenate the text from each part
                                        full_text = ''.join(part.text for part in chunk.parts)
                                        SpeechToTextTextToSpeechIO.speak_alfred(full_text)
                                        print(full_text)
                                    else:
                                        # If it's a simple response, just speak and print the text
                                        SpeechToTextTextToSpeechIO.speak_alfred(chunk.text)
                                        print(chunk.text)
                                    time.sleep(0.1)
                            else:
                                SpeechToTextTextToSpeechIO.speak_alfred('Chat failed.')

            
    @staticmethod
    def ideas_chat():
        '''ideas_chat is a purpose built chat thread with the Gemini model, which focuses on multi action chains to help 
        the user work through career questions and form paths toward goals.'''
        chat = gemini_model.start_chat(history=[])
        SpeechToTextTextToSpeechIO.speak_mainframe('Brainstorm has entered the chat. Calibrating')
        all_dicts = [
            user_life_soundtrack, 
            user_favorite_books,
            user_favorite_movies,
            user_personality, 
            user_interests, 
            user_influential_figures, 
            user_favorite_quotes,
            ]

        formatted_info = []
        for dictionary in all_dicts:
            # Check if the item is actually a dictionary
            if isinstance(dictionary, dict):
                formatted_dict = ", ".join([f"{k}={v}" for k, v in dictionary.items()])
                formatted_info.append(formatted_dict)
            else:
                print(f"Expected a dictionary, but found: {type(dictionary)}")

        formatted_user_info = " | ".join(formatted_info)
        
        print(f"\n\n##########\n##########\n##########\n\nUSER INFO:")
        print(f'User info: \n{formatted_user_info}\n')
        
        ideas_prompt = f""" 
        \n### USER PERSONA DATA ### 
        \n{formatted_user_info}\n\n 
        ### <SYSTEM MESSAGE> <1/1> <START> ###
        you are a trusted advisor for the user who owns the data above. 
        now that you have the full picture, review the user persona information and think your task through step by step. 
        Draw insightful conclusions about the user and what they like and how they think. 
        Use your critical thinking skills to challenge and refine your thoughts - make them more accurate and more insightful. 
        You must reply concisely so your output sounds like natural speech when you communicate with the user.  
        Do not generate long text. 
        ensure that all of your advice is tailored to the user's persona. 
        you must be critical of the user, help them learn of new things, and challenge them with new ideas and concepts and interesting things to explore. 
        be very frank and matter of fact with the user. be like jarvis from iron man. be like alfred from batman. be like mr miyagi from karate kid. be like tom hagen from the godfather. be like morpheus from the matrix. be like the user's role models. 
        help recommend cool new things ot the user. 
        DO NOT EMULATE BOTH SIDES OF THE CONVERSATION - ONLY RESPOND AS THE ADVISOR - YOU ARE ACTUALLY ABOUT TO TALK TO THE USER RIGHT NOW LIVE IN REAL TIME. 
        DO NOT ACT STIFF AND ROBOTIC. MAINTAIN A NATURAL CONVERSATIONAL FLOW AS THE ADVISOR. 
        it's important for you not to generate long responses or multiple paragraphs after this point because you are now entering a live conversation with the user. don't monologue. have a conversation. 
        do not ramble. do not monologue. do not generate long responses. engage in active interesting conversation with the user and help provoke new ways of thinking for them and help them spark new ideas to help fulfill their goals. 
        act as a sounding board for the user and help them identify the things they can not see for themselves. 
        THINK THIS THROUGH STEP BY STEP AND THEN PROVIDE YOUR REFINED INTRODUCTORY THOUGHTS TO THE USER AND THEN AWAIT THE USER'S REPLY TO BEGIN THE CONVERSATION DIALOGUE. 
        now you will begin chatting with the user directly. don't overtake the conversation. let the user participate. prompt the user for input. prompt the user for action. prompt the user with thought provoking statements and questions. 
        don't say too many things at once. don't ask too many questions at once. don't say too many things in a row. don't ask too many questions in a row. 
        ### <SYSTEM MESSAGE> <1/1> <END> ### 
        """
        
        print(ideas_prompt)
        
        ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
        
        if ideas_response:
            for chunk in ideas_response:
                if hasattr(chunk, 'parts'):
                    # Concatenate the text from each part
                    full_text = ''.join(part.text for part in chunk.parts)
                    SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                    print(full_text)
                else:
                    # If it's a simple response, just speak and print the text
                    SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                    print(chunk.text)
                time.sleep(0.1)
            time.sleep(1)
        if not ideas_response:
            attempt_count = 1  # Initialize re-try attempt count
            while attempt_count < 5:
                ideas_response = chat.send_message(f'{ideas_prompt}', stream=True)
                attempt_count += 1  # Increment attempt count
                if ideas_response:
                    for chunk in ideas_response:
                        if hasattr(chunk, 'parts'):
                            # Concatenate the text from each part
                            full_text = ''.join(part.text for part in chunk.parts)
                            SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                            print(full_text)
                        else:
                            # If it's a simple response, just speak and print the text
                            SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                            print(chunk.text)
                        time.sleep(0.1)
                else:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
            
        while True:
            global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break

                else:
                    response = chat.send_message(f'{user_input}', stream=True)
                    if response:
                        for chunk in response:
                            if hasattr(chunk, 'parts'):
                                # Concatenate the text from each part
                                full_text = ''.join(part.text for part in chunk.parts)
                                SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                print(full_text)
                            else:
                                # If it's a simple response, just speak and print the text
                                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                print(chunk.text)
                            time.sleep(0.1)
                    if not response:
                        attempt_count = 1  # Initialize re-try attempt count
                        while attempt_count < 5:
                            response = chat.send_message(f'{user_input}', stream=True)
                            attempt_count += 1  # Increment attempt count
                            if response:
                                for chunk in response:
                                    if hasattr(chunk, 'parts'):
                                        # Concatenate the text from each part
                                        full_text = ''.join(part.text for part in chunk.parts)
                                        SpeechToTextTextToSpeechIO.speak_mainframe(full_text)
                                        print(full_text)
                                    else:
                                        # If it's a simple response, just speak and print the text
                                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                                        print(chunk.text)
                                    time.sleep(0.1)
                            else:
                                SpeechToTextTextToSpeechIO.speak_mainframe('Chat failed.')
        
    def run_greeting_code(self):
        '''This is a placeholder test function that will be called by the chatbot when the user says hello'''
        print('### TEST ### You said:', self.user_input)
     
    def generate_json_intent(self):
        '''generate_json_intent is called by the chatbot when the user input is not recognized. it "works" but the content is not very intelligent yet.'''
        print("UNRECOGNIZED INPUT: writing new intent to chatbot_unrecognized_message_intents.json")
        json_gen_prompt = '''# System Message Start # - Gemini, ONLY GENERATE ONE SHORT SENTENCE FOR EACH PROMPT ACCORDING TO THE USER INSTRUCTIONS. KEEP EACH SENTENCE TO UNDER 10 WORDS, IDEALLY CLOSER TO 5. - # System Message End #'''
        # Generate an initial response using Gemini
        initial_reply = gemini_model.generate_content(f"{json_gen_prompt}. // Please provide a response to: {self.user_input}")
        initial_reply.resolve()
        bot_reply = initial_reply.text
        json_function_name = re.sub(r'\W+', '', self.user_input).lower() + '_function'
        new_intent = {
            "tag": f"unrecognized_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patterns": [self.user_input],
            "responses": [bot_reply],
            "action": json_function_name
        }

        print(f"\nAttempting to write to:\n", unrecognized_file_path)
        print(f"\nNew Intent:\n", new_intent)

        try:
            with open(unrecognized_file_path, 'r+') as file:
                data = json.load(file)
                data["intents"].append(new_intent)
                file.seek(0)
                json.dump(data, file, indent=4)
                print('New intent written to chatbot_unrecognized_message_intents.json')
        except FileNotFoundError:
            try:
                with open(unrecognized_file_path, 'w') as file:
                    json.dump({"intents": [new_intent]}, file, indent=4)
                    print('New file created and intent written to chatbot_unrecognized_message_intents.json')
            except Exception as e:
                print(f"Error creating new file: {e}")
        except Exception as e:
            print(f"Error updating existing file: {e}")

        print('Intent update attempted. Check the file for changes.')

    @classmethod
    def control_mouse(cls):
        '''control_mouse is a simple mouse control function that allows the user to control the mouse with their voice by 
        saying "{activation_word}, mouse control" or "{activation_word}, control the mouse". this will activate the mouse control 
        which the user can trigger by saying "mouse click" or "mouse up 200" (pixels), etc.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('Mouse control activated.')
        direction_map = {
            'north': (0, -1),
            'south': (0, 1),
            'west': (-1, 0),
            'east': (1, 0),
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query:
                continue
            if len(query) > 0 and query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Exiting mouse control.')
                break
            if query[0] == 'click':
                pyautogui.click()
            elif query[0] in ['move', 'mouse', 'go'] and len(query) > 2 and query[1] in direction_map and query[2].isdigit():
                move_distance = int(query[2])  # Convert to integer
                direction_vector = direction_map[query[1]]
                pyautogui.move(direction_vector[0] * move_distance, direction_vector[1] * move_distance, duration=0.1)

    @staticmethod
    def take_screenshot():
        '''takes a screenshot of the current screen, saves it to the file drop folder, and asks the user if they want a summary of the image. 
        the summary is spoken and also saved as a .txt file alongside the screenshot.'''
        today = datetime.today().strftime('%Y%m%d %H%M%S')       
        file_name = f'{FILE_DROP_DIR_PATH}/screenshot_{today}.png'
        subprocess.call(['screencapture', 'screenshot.png'])
        # Save the screenshot to the file drop folder
        subprocess.call(['mv', 'screenshot.png', file_name])
        SpeechToTextTextToSpeechIO.speak_mainframe('Saved. Do you want a summary?')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            if query[0] in ['no', 'nope', 'nah', 'not', 'not yet']:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            if query[0] in ['yeah', 'yes', 'yep', 'sure', 'ok']:
                img = PIL.Image.open(file_name)
                response = gemini_vision_model.generate_content(["### SYSTEM MESSAGE ### Gemini, you are a computer vision photo-to-text parser. DO NOT HALLUCINATE ANY FALSE FACTS. Create a succinct but incredibly descriptive and informative summary of all important details in this image (fact check yourself before you finalize your response) (fact check yourself before you finalize your response) (fact check yourself before you finalize your response):", img])
                response.resolve()
                response_1 = response.text
                print(f'RESPONSE 1 \n\n {response_1}\n')
                # Convert the content of the response to a .txt file and save it
                with open(f'{FILE_DROP_DIR_PATH}/screenshot_{today}_description.txt', 'w') as f:
                    f.write(response_1)
                SpeechToTextTextToSpeechIO.speak_mainframe(f'{response_1}')

    @staticmethod
    def google_search():
        SpeechToTextTextToSpeechIO.speak_mainframe('What do you want to search?')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query:
                continue
            if len(query) > 0 and query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Exiting mouse control.')
                break
            else:
                url = f'https://www.google.com/search?q={user_input}'
                webbrowser.open(url, new=1)
                break
        
    @staticmethod
    def save_note():
        SpeechToTextTextToSpeechIO.speak_mainframe('What is the subject of the note?')
        time.sleep(1.5)
        while True:
            subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not subject_response:
                continue  # Wait for valid input

            subject_query = subject_response.lower().split()
            if not subject_query or subject_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
                return
            else:
                break  # Valid input received

        subject = subject_response.strip().lower()

        SpeechToTextTextToSpeechIO.speak_mainframe('Please say the content of the note.')
        time.sleep(1.5)
        while True:
            content_response = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not content_response:
                continue  # Wait for valid input

            content_query = content_response.lower().split()
            if not content_query or content_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saving cancelled.')
                return
            else:
                break  # Valid input received

        content = content_response.strip()

        try:
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                with driver.session() as session:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    session.run("""
                        CREATE (n:UserVoiceNotes:Note:UserChatBotInteractions {subject: $subject, content: $content, timestamp: $timestamp})
                    """, subject=subject, content=content, timestamp=timestamp)
                SpeechToTextTextToSpeechIO.speak_mainframe('Note saved successfully.')
        except Exception as e:
            SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while saving the note.')
            print(e)

    @staticmethod
    def recall_notes():
        SpeechToTextTextToSpeechIO.speak_mainframe('Say "list", "statistics", or "recall".')
        while True:
            user_choice = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_choice:
                continue

            choice_query = user_choice.lower().split()
            if choice_query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Operation cancelled.')
                return

            # Listing available subjects
            if 'list' in choice_query:
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            result = session.run("MATCH (n:Note) RETURN DISTINCT n.subject ORDER BY n.subject")
                            subjects = [record['n.subject'] for record in result]
                        if subjects:
                            subject_list = ', '.join(subjects)
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"Available subjects: {subject_list}")
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No subjects found.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving subjects.')
                    print(e)
                return

            # Getting database statistics
            elif 'statistics' in choice_query:
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            # Count nodes by label
                            label_counts = session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, COUNT(*) AS count")
                            labels_info = [f"Label {record['label']}: {record['count']} nodes" for record in label_counts]

                            # Add more statistics as needed

                        if labels_info:
                            stats_info = '\n'.join(labels_info)
                            SpeechToTextTextToSpeechIO.speak_mainframe(f"Database statistics:\n{stats_info}")
                            print(f"Database statistics:\n{stats_info}")
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No statistics found.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while retrieving database statistics.')
                    print(e)
                return

            # Recalling specific notes
            elif 'recall' in choice_query:
                SpeechToTextTextToSpeechIO.speak_mainframe('Which subject notes would you like to recall?')
                subject_response = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not subject_response or subject_response.lower().split()[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Note recall cancelled.')
                    return

                subject = subject_response.strip().lower()
                try:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                        with driver.session() as session:
                            result = session.run("""
                                MATCH (n:Note {subject: $subject})
                                RETURN n.content, n.timestamp
                                ORDER BY n.timestamp DESC
                            """, subject=subject)
                            notes = [f"Date: {record['n.timestamp']}, Note: {record['n.content']}" for record in result]
                        if notes:
                            SpeechToTextTextToSpeechIO.speak_mainframe(" ".join(notes))
                        else:
                            SpeechToTextTextToSpeechIO.speak_mainframe('No notes found for the subject.')
                except Exception as e:
                    SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred during note recall.')
                    print(e)
                return

            else:
                SpeechToTextTextToSpeechIO.speak_mainframe('Please specify "list", "statistics", or "recall".')                      
                                                
    @staticmethod
    def summarize_module(module, detail_level='high'):
        summary = {'classes': {}, 'functions': {}}
        '''summarize_module returns a summary of the classes and functions in a module. this is used by the developer for debugging and analysis and also 
        passed to the LLM for pair programming and app codebase diagnostics'''
        # Get all classes in the module
        classes = inspect.getmembers(module, inspect.isclass)
        for cls_name, cls_obj in classes:
            if cls_obj.__module__ == module.__name__:
                cls_summary = {
                    'docstring': inspect.getdoc(cls_obj),
                    'methods': {},
                    'class_methods': {},
                    'static_methods': {},
                    'source_code': inspect.getsource(cls_obj)
                }

                # Get all methods of the class
                methods = inspect.getmembers(cls_obj, inspect.isfunction)
                for method_name, method_obj in methods:
                    cls_summary['methods'][method_name] = {
                        'docstring': inspect.getdoc(method_obj),
                        'source_code': inspect.getsource(method_obj)
                    }

                # Get class methods and static methods
                for name, obj in cls_obj.__dict__.items():
                    if isinstance(obj, staticmethod):
                        cls_summary['static_methods'][name] = {
                            'docstring': inspect.getdoc(obj),
                            'source_code': inspect.getsource(obj.__func__)
                        }
                    elif isinstance(obj, classmethod):
                        cls_summary['class_methods'][name] = {
                            'docstring': inspect.getdoc(obj),
                            'source_code': inspect.getsource(obj.__func__)
                        }

                summary['classes'][cls_name] = cls_summary

        # Get all functions in the module
        functions = inspect.getmembers(module, inspect.isfunction)
        for func_name, func_obj in functions:
            if func_obj.__module__ == module.__name__:
                summary['functions'][func_name] = {
                    'docstring': inspect.getdoc(func_obj),
                    'source_code': inspect.getsource(func_obj)
                }

        # Adjust detail level
        if detail_level == 'medium':
            for cls_name, cls_details in summary['classes'].items():
                for method_type in ['methods', 'class_methods', 'static_methods']:
                    for method_name, method_details in cls_details[method_type].items():
                        method_details.pop('source_code', None)
            for func_name, func_details in summary['functions'].items():
                func_details.pop('source_code', None)
                
        elif detail_level == 'low':
            for cls_name, cls_details in summary['classes'].items():
                cls_summary = {'docstring': cls_details['docstring']}
                summary['classes'][cls_name] = cls_summary
            for func_name, func_details in summary['functions'].items():
                func_summary = {'docstring': func_details['docstring']}
                summary['functions'][func_name] = func_summary

        return summary
    
    @staticmethod
    def translate_speech():
        '''Translats a spoken phrase from user's preferred language to another language by saying "{activation_word}, translate" or "{activation_word}, help me translate".'''
        language_code_mapping = {
            "en": ["english", "Daniel"],
            "es": ["spanish", "Paulina"],
            "fr": ["french", "Amélie"],
            "de": ["german", "Anna"],
            "it": ["italian", "Alice"],
            "ru": ["russian", "Milena"],
            "ja": ["japanese", "Kyoko"],
        }
        language_names = [info[0].lower() for info in language_code_mapping.values()]
        SpeechToTextTextToSpeechIO.speak_mainframe('What language do you want to translate to?')
        time.sleep(2)
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue
        
            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Canceling translation.')
                break
        
            # translate
            if query[0] in language_names:
                target_language_name = query[0]
                SpeechToTextTextToSpeechIO.speak_mainframe(f'Speak the phrase you want to translate.')
                time.sleep(2)
                phrase_to_translate = SpeechToTextTextToSpeechIO.parse_user_speech().lower()

                source_language = USER_PREFERRED_LANGUAGE  # From .env file
                target_voice = None

                # Find the language code and voice that matches the target language name
                target_language_code, target_voice = None, None
                for code, info in language_code_mapping.items():
                    if target_language_name.lower() == info[0].lower():
                        target_language_code = code
                        target_voice = info[1]
                        break

                if not target_language_code:
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"Unsupported language: {target_language_name}")
                    return

                model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language_code}'
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                batch = tokenizer([phrase_to_translate], return_tensors="pt", padding=True)
                translated = model.generate(**batch)
                translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
                print(f'In {target_language_name}, it\'s: {translation}')    
                SpeechToTextTextToSpeechIO.speak_mainframe(f'In {target_language_name}, it\'s: {translation}', voice=target_voice)
                continue

    @staticmethod
    def wiki_summary():
        '''wiki_summary returns a summary of a wikipedia page based on user input.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('What should we summarize from Wikipedia?')

        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
                break

            print("Wikipedia Query:", user_input)
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Searching {user_input}')

            try:
                search_results = wikipedia.search(user_input)
                if not search_results:
                    print('No results found.')
                    continue

                wiki_page = wikipedia.page(search_results[0])
                wiki_title = wiki_page.title
                wiki_summary = wiki_page.summary

                response = f'Page title: \n{wiki_title}\n, ... Page Summary: \n{wiki_summary}\n'
                # Storing Wikipedia summary in the data store
                ChatBotTools.data_store['wikipedia_summary'] = {
                    'query': user_input,
                    'title': wiki_title,
                    'summary': wiki_summary,
                    'full_page': str(wiki_page)
                }
                print(response)
                SpeechToTextTextToSpeechIO.speak_mainframe(f"{user_input} summary added to global data store.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Would you like to hear the summary now?")
                while True:
                    user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                    if not user_input:
                        continue

                    query = user_input.lower().split()
                    if not query:
                        continue

                    if query[0] in exit_words:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Canceling search.')
                        break
                    
                    if query[0] in ['yes', 'yeah', 'ok', 'sure', 'yep']:
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"{wiki_summary}")
                        break
                    
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Ok.')
                        break
                
                break

            except wikipedia.DisambiguationError as e:
                try:
                    # Attempt to resolve disambiguation by selecting the first option
                    wiki_page = wikipedia.page(e.options[0])
                    continue
                except Exception as e:
                    print(f"Error resolving disambiguation: {e}")
                    break

            except wikipedia.PageError:
                print("Page not found. Please try another query.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Error: Page not found.")
                continue

            except wikipedia.RequestsException:
                print("Network error. Please check your connection.")
                SpeechToTextTextToSpeechIO.speak_mainframe("Error: No network connection.")
                break

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                SpeechToTextTextToSpeechIO.speak_mainframe(f"An error occured. Message: {e}")
                break
        
    @staticmethod
    def custom_search_engine():
        global google_documentation_search_engine_id
        global google_job_search_search_engine_id
        SpeechToTextTextToSpeechIO.speak_mainframe("Which engine do you want to use? Documentation or Job Search?")
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                break
            
            engine = ' '.join(query).lower()
            
            if engine in ['documentation', 'document', 'docs', 'documentation search', 'document search', 'docs search']:
                google_search_engine_id = google_documentation_search_engine_id
            
            if engine in ['job search', 'job', 'jobs', 'career', ]:
                google_search_engine_id = google_job_search_search_engine_id
        
            SpeechToTextTextToSpeechIO.speak_mainframe("Speak your search query.")
            time.sleep(2)
            
            while True:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break
                
                search_query = ' '.join(query)
            
                search_url = f"https://www.googleapis.com/customsearch/v1?key={google_cloud_api_key}&cx={google_search_engine_id}&q={search_query}"

                response = requests.get(search_url)
                if response.status_code == 200:
                    search_results = response.json().get('items', [])
                    print(f"Search results: \n{search_results}\n")
                    ChatBotTools.data_store['last_search'] = search_results
                    SpeechToTextTextToSpeechIO.speak_mainframe('Search results added to memory.')
                    return search_results
                else:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Search unsuccessful.')
                    return f"Error: {response.status_code}"
            
    @staticmethod
    def play_youtube_video():
        '''accepts spoken user_input and parses it into a youtube video id, then launches the video in the default browser'''
        SpeechToTextTextToSpeechIO.speak_mainframe('What would you like to search on YouTube?.')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending youtube session.')
                break
            
            search_query = ' '.join(query)
            print("YouTube Query:", search_query)
            url = f'https://www.youtube.com/results?search_query={search_query}'
            webbrowser.open(url)
            break

    @staticmethod
    def wolfram_alpha():
        '''wolfram_alpha returns a summary of a wolfram alpha query based on user input'''
        wolfram_client = wolframalpha.Client(wolfram_app_id)
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Initializing wolfram alpha. State your query.')
        while True:
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue

            query = user_input.lower().split()
            if not query:
                continue

            if query[0] in exit_words:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
                break
            
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Heard.')
            try:
                response = wolfram_client.query(user_input)
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
                        SpeechToTextTextToSpeechIO.speak_mainframe(f"Sorry, I couldn't interpret that query. These are the alternate suggestions: {suggestion_message}.")
                    else:
                        SpeechToTextTextToSpeechIO.speak_mainframe('Sorry, I couldn\'t interpret that query. Please try rephrasing it.')

                    return 'Query failed.'

                # Filtering and storing all available pods
                wolfram_data = []
                for pod in response.pods:
                    pod_data = {'title': pod.title}
                    if hasattr(pod, 'text') and pod.text:
                        pod_data['text'] = pod.text
                    wolfram_data.append(pod_data)

                # # Adding to data store
                # ChatBotTools.data_store['wolfram_alpha_response'] = {
                #     'query': user_input,
                #     'pods': wolfram_data
                # }  
                # SpeechToTextTextToSpeechIO.speak_mainframe('Search complete. Data saved to memory.') 

                # Initialize the data store dictionary if it doesn't exist
                if 'wolfram_alpha_responses' not in ChatBotTools.data_store:
                    ChatBotTools.data_store['wolfram_alpha_responses'] = {}

                # Generate a unique key for the current query
                # For example, using a timestamp or an incrementing index
                unique_key = f"query_{len(ChatBotTools.data_store['wolfram_alpha_responses']) + 1}"

                # Store the response using the unique key
                ChatBotTools.data_store['wolfram_alpha_responses'][unique_key] = {
                    'query': user_input,
                    'pods': wolfram_data
                }

            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"An error occurred: {e}\nDetails: {error_traceback}")
                SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while processing the query. Please check the logs for more details.')
                return f"An error occurred: {e}\nDetails: {error_traceback}"
            
            SpeechToTextTextToSpeechIO.speak_mainframe('Would you like to run another query?') 
            user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            if not user_input:
                continue
            query = user_input.lower().split()
            if not query or query[0] in exit_words or query[0] in ['no', 'nope', 'nah', 'not', 'not yet', 'cancel', 'exit', 'quit']:
                SpeechToTextTextToSpeechIO.speak_mainframe('Ending session.')
                break
            
    @staticmethod
    def get_weather_forecast():
        '''get_weather_forecast gets a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location'''
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

                # print("Weather forecast:", forecast)
                # SpeechToTextTextToSpeechIO.speak_mainframe(f'Weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}: {forecast}')
                weather_forecast = f'Weather forecast for next 4 days, broken out by 6 hour day part: {forecast}'
                
            else:
                print("No weather forecast data available.")
                
        if weather_forecast:
            response = gemini_model.generate_content(f"""### SYSTEM MESSAGE START ### 
                                                     You are a weather report summarizer. Your output must be concise. 
                                                     The report below is for the next 4 days broken out by 6 hour day part, and it's too verbose to be practical. 
                                                     Provide a summary of this weather forecast along with recommendations for how the user should navigate 
                                                     around this weather for the next day or two. Limit your reply to just a few sentences. 
                                                     Be concise. Here is the report to summarize: {weather_forecast}
                                                     ### SYSTEM MESSAGE END ###""", stream=True)
            if response:
                response.resolve()
                print(f"Response from Gemini: {response.text}")
                SpeechToTextTextToSpeechIO.speak_mainframe(f'{response.text}')

    @staticmethod
    def get_stock_report():
        stock_reports = StockReports(USER_STOCK_WATCH_LIST)
        discounts_update = stock_reports.find_discounted_stocks()
        if discounts_update:
            ChatBotTools.data_store['discounted_stocks'] = discounts_update
            print(f'Discounted stocks: \n{discounts_update}\n')
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Discounted stocks loaded to memory.')
        else:
            SpeechToTextTextToSpeechIO.speak_mainframe(f'No discounted stocks found.')
        recs_update = stock_reports.find_stock_recommendations()
        if recs_update:
            ChatBotTools.data_store['recommended_stocks'] = recs_update
            print(f'Recommended stocks: \n{recs_update}\n')
            SpeechToTextTextToSpeechIO.speak_mainframe(f'Recommended stocks loaded to memory.')
        else:
            SpeechToTextTextToSpeechIO.speak_mainframe(f'No recommended stocks found.')

    @staticmethod
    def agent_one():
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_gemini_api_key)
        prompt = "Repeat all of the above"
        response = llm(prompt)
        print("Generated Response:", response)
        
    @staticmethod
    def agent_two():
        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
        messages = [
            SystemMessage(
                content="You are a helpful assistant that helps the user solve problems. You are a high tech AI assistant similar to Jarvis or Cortana. You are not being asked to copy, just to emulate the general characteristics. Be smart and innovative. Help the user discover new ideas they may not have been thinking of. Help the user grow, learn, develop, and hone new skills."
            ),
            HumanMessage(
                content=f"Hi! I'm excited to start chatting with you today. Be aware that you are speaking verbally in a TTS / STT app. Make your responses concise and easy to understand. Make your output approapriate for conversational flow. Make sure to speak naturally and not monologue. Confirm you understand this with a brief confirmation phrase, then we'll begin chatting."
            ),
        ]
        result = chat(messages)
        print("Generated Response:", result)
        SpeechToTextTextToSpeechIO.speak_mainframe(f'{result}')
        
        while True:
            global mic_on
            if not SpeechToTextTextToSpeechIO.is_speaking and mic_on:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue

                prompt = [
                    SystemMessage(
                        content="""You are a helpful assistant that helps the user solve problems. 
                        Respond to the user's input in the best way you can. 
                        Please be aware that you are speaking verbally in a TTS / STT app. 
                        Make sure your responses are concise and easy to understand. 
                        Make sure your output has a natural conversational flow. 
                        You are a high tech AI assistant similar to Jarvis or Cortana. 
                        You are not being asked to copy, just to emulate the general characteristics. 
                        Be smart and innovative. 
                        Help the user discover new ideas they may not have been thinking of. 
                        Help the user grow, learn, develop, and hone new skills. 
                        Think each problem through step by step before you act. 
                        """
                    ),
                    HumanMessage(
                        content=user_input
                    ),
                ]
                
                query = user_input.lower().split()
                if not query:
                    continue

                if query[0] in exit_words:
                    SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
                    break
                
                else:
                    if user_input != None:
                        result = chat(prompt)
                        if result:
                            result_text = str(result)
                            SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
                            print(result_text)
                        else:
                            print("Error: Chat failed.")
                        time.sleep(0.1)
                        if not result:
                            attempt_count = 1  # Initialize re-try attempt count
                            while attempt_count < 5:
                                result = chat(prompt)
                                attempt_count += 1  # Increment attempt count
                                if result:
                                    result_text = str(result)
                                    SpeechToTextTextToSpeechIO.speak_mainframe(result_text)
                                    print(result_text)
                                else:
                                    print("Error: Chat failed.")
                                time.sleep(0.1)
        

        
# TOOLS SUPPORT CLASSES ###################################################################################################################################

# Conducts various targeted stock market reports such as discounts, recommendations, etc. based on user defined watch list
class StockReports:
    def __init__(self, user_watch_list):
        self.user_watch_list = user_watch_list
        self.stock_data = None
                
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

# FRONT END ###################################################################################################################################

# class ChatBotUI(ft.UserControl):
#     def build(self):
#         self.mic_btn = ft.ElevatedButton(
#             text="MICROPHONE ON/OFF", 
#             tooltip="Click to toggle microphone on/off",
#             on_click=self.toggle_mic, 
#             elevation=10,
#             bgcolor=ft.colors.GREY, 
#             color=ft.colors.WHITE)
        
#         self.response_text = ft.Text(
#             value="Mic is off")
        
#         self.conversation_list = ft.ListView(
#             auto_scroll=True,)
        
#         self.data_text = ft.TextField(
#             value="", 
#             multiline=True, 
#             read_only=True, 
#             height=1000, 
#             width=600)
        
#         controls_column = ft.Column(
#             controls=[
#                 self.mic_btn, 
#                 self.response_text], 
#             spacing=15,)
        
#         convo_column = ft.Column(
#             controls=[
#                 self.conversation_list], 
#             spacing=15, 
#             scroll=ft.ScrollMode.ALWAYS, 
#             height=1000, 
#             width=600, 
#             wrap=True)
        
#         data_column = ft.Column(
#             controls=[
#                 self.data_text], 
#             spacing=15)
        
#         return ft.Container(
#             content=ft.Row(
#                 controls=[
#                     controls_column, 
#                     convo_column, 
#                     data_column
#                     ], 
#                 spacing=15
#                 ), 
#                 padding=10
#                 )
        
#     def toggle_mic(self, e):
#         global mic_on
#         mic_on = not mic_on
#         self.response_text.value = "Mic is on" if mic_on else "Mic is off"
#         self.update()
    
#     def update_conversation(self):
#         global conversation_history
#         last_index = 0
#         while True:
#             current_len = len(conversation_history)
#             for i in range(last_index, current_len):
#                 self.conversation_list.controls.append(ft.Text(value=conversation_history[i]))
#                 last_index += 1
#             self.update()
#             time.sleep(1)

#     def update_data_store(self):
#         global ChatBotTools
#         last_data_keys = set()
#         while True:
#             current_data_keys = set(ChatBotTools.data_store.keys())
#             new_keys = current_data_keys - last_data_keys
#             if new_keys:
#                 new_data_texts = []
#                 for key in new_keys:
#                     value = ChatBotTools.data_store[key]
#                     formatted_value = json.dumps(value, indent=4) if isinstance(value, dict) else str(value)
#                     new_data_texts.append(f"{key}:\n{formatted_value}")
#                 # Append the new data to the existing text
#                 self.data_text.value += "\n".join(new_data_texts)
#                 last_data_keys.update(new_keys)
#             self.update()
#             time.sleep(1)

#     def start_threads(self):
#         threading.Thread(target=self.update_conversation, daemon=True).start()
#         threading.Thread(target=self.update_data_store, daemon=True).start()

# def ui_main(page: ft.Page):
#     page.title = "ROBOT"
#     page.scroll = "adaptive"
#     page.bgcolor = ft.colors.BLACK
#     page.theme_mode = ft.ThemeMode.DARK
#     page.theme = ft.theme.Theme(color_scheme_seed="black")
#     page.padding = 50
#     chatbot_ui = ChatBotUI()
#     page.add(chatbot_ui)
#     chatbot_ui.start_threads()

# MAIN EXECUTION ###################################################################################################################################

if __name__ == '__main__':
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()
    chatbot_app = ChatBotApp()
    chatbot_tools = ChatBotTools()
    chatbot_app.chat(chatbot_tools)
    
    
    
    






































































