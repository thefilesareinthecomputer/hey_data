# chatbot_app_with_tools.py

# IMPORTS ###################################################################################################################################

# standard imports
from datetime import datetime
from dotenv import load_dotenv
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
# third party imports
from nltk.stem import WordNetLemmatizer
from transformers import MarianMTModel, MarianTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
import certifi
import google.generativeai as genai
import numpy as np
import nltk
import pyautogui
import requests
import speech_recognition as sr
import tensorflow as tf
import wikipedia
import wolframalpha
# local imports
# from chatbot_training import train_chatbot_model
# train_chatbot_model()

# CONSTANTS ###################################################################################################################################

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
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context
context = create_ssl_context()
print("SSL Context Details:")
print(f"CA Certs File: {context.cert_store_stats()}")
print(f"Protocol: {context.protocol}")
print(f"Options: {context.options}")
print(f"Verify Mode: {context.verify_mode}")
print(f"Verify Flags: {context.verify_flags}")
print(f"Check Hostname: {context.check_hostname}")
print(f"CA Certs Path: {certifi.where()}")
print("\n\n")

# Set API keys and other sensitive information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_documentation_search_engine_id = os.getenv('GOOGLE_DOCUMENTATION_SEARCH_ENGINE_ID')
print('API keys and other sensitive information loaded from environment variables.\n\n')

# Establish the TTS bot's wake/activation word and script-specific global constants
activation_word = os.getenv('ACTIVATION_WORD', 'robot')
password = os.getenv('PASSWORD', 'None')
exit_words = os.getenv('EXIT_WORDS', 'None').split(',')
print(f'Activation word is {activation_word}\n\n')

# Initialize the Google Gemini LLM
genai.configure(api_key=google_gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-pro')  
print('Google Gemini LLM initialized.\n\n') 

lemmmatizer = WordNetLemmatizer()
intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())
words = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'rb'))
classes = pickle.load(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'rb'))
chatbot_model = tf.keras.models.load_model(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_model.keras')
unrecognized_file_path = f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_unrecognized_message_intents.json'
print('Local chatbot model loaded.\n\n')

# CLASS DEFINITIONS ###################################################################################################################################

class SpeechToTextTextToSpeechIO:
    '''the SpeechToTextTextToSpeechIO class handles the speech to text and text to speech functionality of the chatbot. It also handles the speech output queue.
    the speech output queue operates on a first in first out basis. The speech manager thread is constantly checking the queue for new items. 
    the speech manager thread will not start a new speech output until the previous speech output has finished. 
    the speech manager runs on its own thread so that the bot can still recieve input while speaking. this hasn't been built out to its full potential yet 
    because we havne't figured out how to get the bot to listen for input while it is speaking without hearing itself. we also sometimes have issues 
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
                            # words = text.split()
                            # chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
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
        words = text.split()
        number_of_words = len(words)
        minutes = number_of_words / rate
        seconds = minutes * 60
        return seconds + 1
    
    @classmethod
    def speak_mainframe(cls, text, rate=190, chunk_size=1000, voice=USER_PREFERRED_VOICE):
        '''speak_mainframe contains the bot's speech output voice settings, and it puts each chunk of text output from the bot or the LLM 
        into the speech output queue to be processed in sequential order. it also separately returns the estimated duration of the speech 
        output (in seconds), using thecalculate_speech_duration function.'''
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
    
    # Shuffle the training data
    random.shuffle(training)

    # Split the training data into X and Y
    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])

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
        '''get_response returns a response from the chatbot based on the most likely match for 
        predicted class (subject tag) of the user input and returns a random response from the predetermined 
        list of responses for that subject tag class in the chatbot_intents.json file (the training data). 
        these JSON objects also contain the names of the functions that are called when the user input matches: 
        
        {
            "intents": [
        {
        "tag": "conversation_unrecognized",
        "patterns": [""],
        "responses": ["Can you re-phrase that?", 
            "What was that?",
            "I didn't understand that.",
            "I don't understand that."],
        "action": "generate_json_intent"
        },
        {
        "tag": "conversation_greeting",
        "patterns": ["hello there",
            "hey how is it going",
            "hi my name is",
            "hello", "hi",
            "whats up",
            "aloha",
            "howdy",
            "yo",
            "hey",
            "hi there",
            "hey there",
            "sup",
            "wassup",
            "hey nice to meet you"],
        "responses": ["Hi. How can I help you?",
            "Hello. How can I help you?",
            "Hi. How's it going?",
            "Hello there. How can I help?", 
            "Hey. Whats up?", 
            "Hey. Can I help with something?"],
        "action": "run_greeting_code"
        },
        {
        "tag": "conversation_capabilities",
        "patterns": ["tell me what you know", 
            "tell me which questions you understand", 
            "tell me what you can do", 
            "what can you do?",
            "I need help programming",
            "Can you assist me with coding",
            "I'm not sure what to do with this python program",
            "Help me out with my python code in this program",
            "what functions do you have",
            "what can this code do",
            "what does this code know how to do",
            "what does this code understand",
            "tell me how to use this app program",
            "what does this app know how to do", 
            "what kind of things does this code do", 
            "what questions does the app in this code understand"],
        "responses": ["Scanning the codebase..."],
        "action": "describe_capabilities"
        },
        '''
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
        '''chat is the main chatbot function. it takes user input, predicts the class (subject tag) of the user input, 
        and returns a response from the chatbot with the get_response function based on the most likely match from the predict_class function. 
        if the JSON intent for the matched response contains a function name in it's 'action' key, the function is called. 
        the function name is then used to call the function from the ChatBotTools class.'''
        print('Start talking with the bot (type quit to stop)!')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Online.')
        
        while True:
            if not SpeechToTextTextToSpeechIO.is_speaking:
                user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                if not user_input:
                    continue
                
                if user_input:
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
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.summarizer = pipeline("summarization", model=self.model, legacy=False, tokenizer=self.tokenizer)
        self.user_input = None  
        
    def set_user_input(self, input_text):
        '''Sets the user input for use in other methods.'''
        self.user_input = input_text

    def run_greeting_code(self):
        '''This is a placeholder test function that will be called by the chatbot when the user says hello'''
        print('### TEST ### You said:', self.user_input)
               
    def generate_variations(self, text):
        '''generate_variations generates variations of the user input and bot reply using the T5 summarization model. 
        this is very rudimentary and needs to be improved. the goal is to generate variations of the user input and bot reply for fabricating 
        new intents for unrecognized bot interactions to add to the chatbot_intents.json training data file. this is a work in progress.'''
        summaries = self.summarizer(text, max_length=45, min_length=15, do_sample=False)
        return [summary['summary_text'] for summary in summaries]
     
    def generate_json_intent(self):
        '''generate_json_intent is called by the chatbot when the user input is not recognized. it "works" but the content is not very intelligent yet.'''
        print("UNRECOGNIZED INPUT: writing new intent to chatbot_unrecognized_message_intents.json")

        json_gen_prompt = '''# System Message Start # - Gemini, ONLY GENERATE ONE SHORT SENTENCE FOR EACH PROMPT ACCORDING TO THE USER INSTRUCTIONS. KEEP EACH SENTENCE TO UNDER 10 WORDS, IDEALLY CLOSER TO 5. - # System Message End #'''
        # Generate an initial response using Gemini
        initial_reply = gemini_model.generate_content(f"{json_gen_prompt}. // Please provide a response to: {self.user_input}")
        initial_reply.resolve()
        bot_reply = initial_reply.text

        # Generate variations of user input and bot reply
        user_input_variations = self.generate_variations(self.user_input)
        bot_reply_variations = self.generate_variations(bot_reply)

        # Simple function name generation
        json_function_name = re.sub(r'\W+', '', self.user_input).lower() + '_function'

        new_intent = {
            "tag": f"unrecognized_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patterns": [self.user_input] + user_input_variations,
            "responses": [bot_reply] + bot_reply_variations,
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

    @staticmethod
    def describe_capabilities():
        '''describe_capabilities delivers a summary of the codebase along with a context explanation prompt template 
        to the Gemini model for interpretation and pair programming.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('Reading the code... One sec...')
        chat = gemini_model.start_chat(history=[])
        diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__])
        print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
        prompt = f'''### SYSTEM MESSAGE ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
        Here is a summary of their Python codebase. The user is going to ask you to describe the capabilities of their codebase.: 
        \n {diagnostic_summary}\n
        ### SYSTEM MESSAGE ### Gemini, please read and deeply unterstand all the methods, static methods, and class methods in 
        this codebase - examine this code and think about the functionalities, strengths, opportunities for improvement, 
        design principles that have been applied, etc. Also assess the skill level of the developer, and 
        think of helpful advice for them and how they can improve their app and the overall cohesiveness and optimization of their code. 
        Be like a coach for the developer. You are a senior architect who is mentoring them by reviewing 
        their code and helping them improve their app and their overall software development skills. 
        Refer to the user directly in the second person tense. You are interacting with the developer directly. 
        Read the code, and then say "I've read the code. What do you want to discuss first?", and then await further instructions. 
        ## wait for user input after you acknowledge this message ##'''
        diagnostic_response = chat.send_message(f'{prompt}', stream=True)
        if diagnostic_response:
            for chunk in diagnostic_response:
                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                time.sleep(0.1)
                print(chunk.text)
                print('\n')
            time.sleep(1)     
        
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
                                             
            else:
                response = chat.send_message(f'{user_input}', stream=True)
                if response:
                    for chunk in response:
                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                        time.sleep(0.1)
        
    @staticmethod
    def gemini_chat():
        '''gemini_chat is a more general purpose chat thread with the Gemini model, with an optional branch for 
        running thorough diagnostics on the app codebase.'''
        SpeechToTextTextToSpeechIO.speak_mainframe('Initializing...')
        chat = gemini_model.start_chat(history=[])
        
        prompt_template = '''Gemini, you are in a verbal chat with the user via a 
        STT / TTS application. Please generate text that sounds like natural speech 
        rather than written text. Please avoid monologuing or including anything in the output that will 
        not sound like natural spoken language. After confirming you understand this message, the chat will proceed. 
        Refer to the user directly in the second person tense. You are talking to them directly. 
        Please confirm your understanding of these instructions by simply saying "Chat loop is open" 
        and then await another prompt from the user. ## wait for user input after you acknowledge this message ##'''

        intro_response = chat.send_message(f'{prompt_template}', stream=True)
        
        if intro_response:
            for chunk in intro_response:
                SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                time.sleep(0.1)
        
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
            
            if query[0] == 'access' and query [1] == 'global' and query[2] == 'memory':
                SpeechToTextTextToSpeechIO.speak_mainframe('Accessing global memory.')
                data_store = ChatBotTools.data_store
                print(ChatBotTools.data_store)
                data_prompt = f'''### SYSTEM MESSAGE ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                Here is the data they've pulled into the conversation so far. The user is going to ask you to discuss this data: 
                \n {data_store}\n
                ### SYSTEM MESSAGE ### Gemini, please read and deeply unterstand all the data and metadata in 
                this dictionary - examine this data and plase it all into context together. 
                Refer to the user directly in the second person tense. You are conversing with the user directly. 
                Read the data, and then say "I've read the data. What do you want to discuss first?", and then await further instructions. 
                ## wait for user input after you acknowledge this message ##'''
                data_response = chat.send_message(f'{data_prompt}', stream=True)
                if data_response:
                    for chunk in data_response:
                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                        time.sleep(0.1)
                        print(chunk.text)
                        print('\n')
                    time.sleep(1)
                continue
                                             
            else:
                response = chat.send_message(f'{user_input}', stream=True)
                if response:
                    for chunk in response:
                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                        time.sleep(0.1)
            
            if query[0] == 'diagnostics':
                SpeechToTextTextToSpeechIO.speak_mainframe('Running diagnostics.')
                diagnostic_summary = ChatBotTools.summarize_module(sys.modules[__name__])
                print(f'DIAGNOSTIC SUMMARY: \n\n{diagnostic_summary}\n\n')
                prompt = f'''### SYSTEM MESSAGE ### Gemini, the user is currently speaking to you from within their TTS / STT app. 
                Here is a summary of their Python code. The user is going to ask you to describe the capabilities of their codebase.: 
                \n {diagnostic_summary}\n
                ### SYSTEM MESSAGE ### Gemini, please read and deeply unterstand all the methods, static methods, and class methods in 
                this codebase - examine this code and think about the functionalities, strengths, opportunities for improvement, 
                design principles that have been applied, etc. Also assess the skill level of the developer, and 
                think of helpful advice for them and how they can improve their app and the overall cohesiveness and optimization of their code. 
                Be like a coach for the developer. You are a senior architect who is mentoring them by reviewing 
                their code and helping them improve their app and their overall software development skills. 
                Refer to the user directly in the second person tense. You are interacting with the developer directly. 
                Read the code, and then say "I've read the code. What do you want to discuss first?", and then await further instructions. 
                ## wait for user input after you acknowledge this message ##'''
                diagnostic_response = chat.send_message(f'{prompt}', stream=True)
                if diagnostic_response:
                    for chunk in diagnostic_response:
                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                        time.sleep(0.1)
                        print(chunk.text)
                        print('\n')
                    time.sleep(1)
                continue
                                             
            else:
                response = chat.send_message(f'{user_input}', stream=True)
                if response:
                    for chunk in response:
                        SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
                        time.sleep(0.1)

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
    def summarize_module(module):
        '''summarize_module returns a summary of the classes and functions in a module. this is used by the developer for debugging and analysis and also 
        passed to the LLM for pair programming and app codebase diagnostics'''
        summary = {
            'classes': {},
            'functions': {},
        }

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
        '''wiki_summary returns a summary of a wikipedia page based on user input by saying 
        "{activation_word}, wiki summary" or "{activation_word}, wikipedia summary".'''
        SpeechToTextTextToSpeechIO.speak_mainframe(f'What should we summarize from Wikipedia?')
        time.sleep(1)
        wikipedia_summary_query = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
        print("Wikipedia Query:", wikipedia_summary_query)  
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Searching {wikipedia_summary_query}')
        search_results = wikipedia.search(wikipedia_summary_query)
        if not search_results:
            print('No results found.')
        try:
            wiki_page = wikipedia.page(search_results[0])
        except wikipedia.DisambiguationError as e:
            wiki_page = wikipedia.page(e.options[0])
        wiki_title = str(wiki_page.title)
        wiki_summary = str(wiki_page.summary)
        response = f'Page title: \n{wiki_title}\n, ... Page Summary: \n{wiki_summary}\n'
        # Storing Wikipedia summary in the data store
        ChatBotTools.data_store['wikipedia_summary'] = {
            'query': wikipedia_summary_query,
            'title': wiki_title,
            'summary': wiki_summary,
            'full_page': str(wiki_page)
        }
        print(response)
        SpeechToTextTextToSpeechIO.speak_mainframe("Wikipedia summary added to global memory store.")
        # SpeechToTextTextToSpeechIO.speak_mainframe(response)
        # context_prompt = f'''User query: \n {wikipedia_summary_query}\n, .. Page title: \n{wiki_title}\n, ... 
        # Page Summary: \n{wiki_summary}\n, ... Full Wikipedia Page: \n{wiki_page}\n. 
        # \n\n ### SYSTEM MESSAGE ### Gemini, please read the Wikipedia page and then summarize it in your own words. 
        # \n\n ## wait for user input after your short summary ##'''
        # chat = gemini_model.start_chat(history=[])
        
        # intro_response = chat.send_message(f'{context_prompt}', stream=True)
                
        # if intro_response:
        #     for chunk in intro_response:
        #         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
        #         time.sleep(0.1)
        
        # while True:
        #     user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
        #     if not user_input:
        #         continue

        #     query = user_input.lower().split()
        #     if not query:
        #         continue

        #     if query[0] in exit_words:
        #         SpeechToTextTextToSpeechIO.speak_mainframe('Ending chat.')
        #         break
                                            
        #     else:
        #         response = chat.send_message(f'{user_input}', stream=True)
        #         if response:
        #             for chunk in response:
        #                 SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
        #                 time.sleep(0.1)        
        
    @staticmethod
    def wolfram_alpha():
        '''wolfram_alpha returns a summary of a wolfram alpha query based on user input'''
        wolfram_client = wolframalpha.Client(wolfram_app_id)
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Initializing wolfram alpha. State your query.')
        wolfram_alpha_query = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Heard. Calculating.')
        try:
            response = wolfram_client.query(wolfram_alpha_query)
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
                SpeechToTextTextToSpeechIO.speak_mainframe(response_text)
                print(f'User: {wolfram_alpha_query} \nWolfram|Alpha: {response_text}')

            else:
                SpeechToTextTextToSpeechIO.speak_mainframe("I found no information in the specified categories.")

            # Asking user for interest in other pods
            for pod in response.pods:
                if pod.title not in relevant_pods_titles:
                    SpeechToTextTextToSpeechIO.speak_mainframe(f"Do you want to hear more about {pod.title}? Say 'yes' or 'no'.")
                    user_input = SpeechToTextTextToSpeechIO.parse_user_speech().lower()
                    if user_input == 'yes' and hasattr(pod, 'text') and pod.text:
                        SpeechToTextTextToSpeechIO.speak_mainframe(pod.text)
                        continue
                    elif user_input == 'no':
                        break

            return response_text        

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"An error occurred: {e}\nDetails: {error_traceback}")
            SpeechToTextTextToSpeechIO.speak_mainframe('An error occurred while processing the query. Please check the logs for more details.')
            return f"An error occurred: {e}\nDetails: {error_traceback}"
        
        
        
        

        
        
        













# # Get a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location
# def get_weather_forecast():
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






if __name__ == '__main__':
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()
    chatbot_app = ChatBotApp()
    chatbot_tools = ChatBotTools()
    chatbot_app.chat(chatbot_tools)
    
    
    
    
    
    






'''

########## SYSTEM MESSAGE ##########
# Hi there! Today you'll be helping the user write some new code.
# The user will direct which code you should write and how, then the user will execute it in their environment.
# Language: python
# Version: 3.11.4
# Project: adding additional features to a chatbot_app.py module of a chatbot app.
# requirement: move the ChatBotApp and ChatBotTools into classes (already done).
# requirement: implement function calling into the chatbot_app.py module, so that the chatbot can execute functions when they are clled by their corresponding matches from the chatbot_intents.json file which currently only contains placeholder comments in the action field rather than actual functions or function names.
# instructions: generate the python code required to meet these requirements so the user can add it to the module below.
# instructions: make sure all your code is complete and will work for the user in their environment.
# instructions: don't use any placeholders in your code, make sure it is all complete and ready to go.
# instructions: if you don't have enough info and need to ask the user a question, just ask it in the chat.
# instructions: explain to the user how/where to implement the code, but primarily focus on writing the code.

########## ASSOCIATED CODE FOR CONTEXT ##########

# JSON training data example from the chatbot_intents.json file:

{
    "intents": [
{
  "tag": "conversation_unrecognized",
  "patterns": [""],
  "responses": ["Can you re-phrase that?", 
    "What was that?",
    "I didn't understand that.",
    "I don't understand that."],
  "action": "generate_json_intent"
},
{
  "tag": "conversation_greeting",
  "patterns": ["hello there",
    "hey how is it going",
    "hi my name is",
    "hello", "hi",
    "whats up",
    "aloha",
    "howdy",
    "yo",
    "hey",
    "hi there",
    "hey there",
    "sup",
    "wassup",
    "hey nice to meet you"],
  "responses": ["Hi. How can I help you?",
    "Hello. How can I help you?",
    "Hi. How's it going?",
    "Hello there. How can I help?", 
    "Hey. Whats up?", 
    "Hey. Can I help with something?"],
  "action": "run_greeting_code"
},
{
  "tag": "conversation_capabilities",
  "patterns": ["tell me what you know", 
    "tell me which questions you understand", 
    "tell me what you can do", 
    "what can you do?",
    "I need help programming",
    "Can you assist me with coding",
    "I'm not sure what to do with this python program",
    "Help me out with my python code in this program",
    "what functions do you have",
    "what can this code do",
    "what does this code know how to do",
    "what does this code understand",
    "tell me how to use this app program",
    "what does this app know how to do", 
    "what kind of things does this code do", 
    "what questions does the app in this code understand"],
  "responses": ["Scanning the codebase..."],
  "action": "describe_capabilities"
},
# continued...

# Neural network architecture in the chatbot_training.py module:

from dotenv import load_dotenv
import json
import os
import pickle
import random

from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import tensorflow as tf

load_dotenv()
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

lemmatizer = WordNetLemmatizer()

intents = json.loads(open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_words.pkl', 'wb'))
pickle.dump(classes, open(f'{PROJECT_ROOT_DIRECTORY}/src/src_local_chatbot/chatbot_classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the training data into X and Y
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

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
print("Done!")

# chatbot_app.py module (the one we are currently working on):

'''






























    
# # Main class for custom search engines in Google Cloud, etc. and a static method which engages a chatbot wrapper around the engines
# class CustomSearchEngines:
#     def __init__(self):
#         self.engines = {
#             'documentation': {
#                 'api_key': google_cloud_api_key,
#                 'cse_id': google_documentation_search_engine_id
#             }
#             # Add more search engines here if needed
#         }

#     def google_custom_search_engine(self, engine_name, query):
#         if engine_name not in self.engines:
#             print(f"Search engine {engine_name} not found.")
#             return None

#         engine = self.engines[engine_name]
#         url = "https://www.googleapis.com/customsearch/v1"
#         params = {
#             'key': engine['api_key'],
#             'cx': engine['cse_id'],
#             'q': query
#         }

#         try:
#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             return response.json()
#         except requests.RequestException as e:
#             print(f"Error in {engine_name} Search: {e}")
#             return None

#     @staticmethod
#     def search_chat_bot():
#         search_engines = CustomSearchEngines()
#         engine_name = ''
#         search_results = [] 
#         chat = model.start_chat(history=[])
#         SpeechToTextTextToSpeechIO.speak_mainframe("Specify the search engine to use.")
#         while not engine_name:
#             input = SpeechToTextTextToSpeechIO.parse_user_speech()
#             if input:
#                 engine_name = input.lower()
#                 print(f"Search Assistant Active using {engine_name}.")
#                 time.sleep(.5)
#             if not input:
#                 continue
#             else:
#                 continue
                            
#         SpeechToTextTextToSpeechIO.speak_mainframe("Please say what you'd like to search")
        
#         while True:
#             user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
            
#             if not user_input:
#                 continue
            
#             query = user_input.lower()
            
#             if query == 'exit search':
#                 break
            
#             results = search_engines.google_custom_search_engine(engine_name, query)
            
#             if results and results.get('items'):
#                 for item in results['items']:
#                     search_results.append(item)  # Storing the entire item
#                     print(f"RESULT: {item}\n\n")

#                 prompt_template = chat.send_message(
#                     '''# System Message # - Gemini, you are in a verbal chat with the user via a 
#                     STT / TTS application. Please generate your text in a way that sounds like natural speech 
#                     when it's spoken by the TTS app. Please avoid monologuing or including anything in the output that will 
#                     not sound like natural spoken language. After confirming you understand this message, the chat will proceed. Please 
#                     confirm your understanding of these instructions by simply saying "Chat loop is open."''', 
#                     stream=True)

#                 if prompt_template:
#                     for chunk in prompt_template:
#                         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
#                         time.sleep(.25)
#                     time.sleep(1)
                    
#                 search_analysis = chat.send_message(
#                     f'''Hi Gemini. The user just engaged a Google custom 
#                     search engine with this query: "{query}". 
#                     These are the search results: {search_results}. 
#                     Please analyze the results while interpreting the true meaning of 
#                     the user's query, then also apply your own internal knowledge. 
#                     Please guide the user in how to solve the problem or answer the question in their query. 
#                     If necessary, also guide the user in crafting a more efficient and effective query. 
#                     Please help guide the user in the right direction. 
#                     Your output should be suitable for a verbal chat TTS app that sounds like natural spoken language. 
#                     Please keep your answers short, direct, and concise. Thank you!''', 
#                     stream=True)
                
#                 if search_analysis:
#                     for chunk in search_analysis:
#                         SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
#                         time.sleep(.25)
#                     time.sleep(1)
                    
#                 while True:
#                     user_input = SpeechToTextTextToSpeechIO.parse_user_speech()
                    
#                     if not user_input:
#                         continue
                    
#                     query = user_input.lower().split()
                    
#                     if query[0] == activation_word and query[1] == 'new' and query[2] == 'search':
#                         SpeechToTextTextToSpeechIO.speak_mainframe("Please say what you'd like to search.")
#                         time.sleep(1)
#                         break
#                     else:
#                         response = chat.send_message(f'{user_input}', stream=True)
#                         if response:
#                             for chunk in response:
#                                 SpeechToTextTextToSpeechIO.speak_mainframe(chunk.text)
#                             time.sleep(1)
                                
#             else:
#                 SpeechToTextTextToSpeechIO.speak_mainframe("No results found or an error occurred.")
    
    




































