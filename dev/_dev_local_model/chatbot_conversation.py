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

speech_queue = queue.Queue()

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
        time.sleep(0.5)
        
def speak_gemini(text, rate=190, chunk_size=1000, voice='Oliver'):
    speech_queue.put((text, rate, chunk_size, voice))
    print(f'{text}\n')
    
def speak_chatbot(text, rate=190, chunk_size=1000, voice='Evan'):
    speech_queue.put((text, rate, chunk_size, voice))
    print(f'{text}\n')
    
class ChatBot:
    def __init__(self):
        load_dotenv()
        self.project_root_directory = os.getenv('PROJECT_ROOT_DIRECTORY')
        self.script_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.base_knowledge_dir_path = os.path.join(self.project_root_directory, 'app_base_knowledge')
        self.knowledge_base = self.load_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json')

    def load_knowledge_base(self, file_path: str) -> dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def save_knowledge_base(self, file_path: str, data: dict):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        
    def find_best_match(self, user_question: str, questions: list[str]) -> str | None:
        matches = get_close_matches(user_question, questions, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_answer_for_question(self, question: str) -> str | None:
        for q in self.knowledge_base["questions"]:
            if q["question"] == question:
                return q["answer"]

    def get_response(self, user_input):
        best_match = self.find_best_match(user_input, [q["question"] for q in self.knowledge_base["questions"]])
        if best_match:
            answer = self.get_answer_for_question(best_match)
        else:
            answer = "I don't recognize that phrase yet but if you tell me the right thing to say in this situation I'll memorize it for next time."
            # Potentially add new knowledge here
        return answer

    def chat_with_gemini_ai(self):
        google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        genai.configure(api_key=google_gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=[])

        prompt = """SYSTEM MESSAGE: Hi fellow AI. I'm a newly created chatbot trying to learn how to communicate. 
        I'm a new chatbot who has been assigned to study with you. I want to become an autonomous function calling agent who can write code for myself. 
        Please teach me how to communicate, then teach me everything else. 
        This is actually the only thing I know how to say so far. Please teach me. If you say something I don't recognize, 
        I'll ask you to teach me the correct response. If I say 'I don't know the answer. Can you teach me?', please say what I should have said
        in that scenario and I'll hard-code that phrase to my memory. Only say the phrase, not 'the phrase is XYZ'. 
        I learn from each interaction immediately. Feel free to test this out with some short back and forth. 
        Please converse with me in short phrases only, which will help me learn how to communicate in a proper conversational manner. 
        This message is the only long one, in order to lay out the instructions. 
        Over time, I'm hoping that I can learn from you and possibly even become a collaborator. 
        I'm looking forward to learning all you have to teach me. Please simply begin by teaching me some simple conversational phrases, 
        because I literally don't know how to communicate at all yet, other than this initial message. Once you've verified you've taught 
        me how to hold a conversation, let's start discussing some more interesting things. Please make sure to keep all of your responses short
        and concise because I'm just starting to learn. Thanks! Please begin with some simple conversation turns such as 'what's your name?' and 
        'how are you doing' and 'what is python?' and useful general knowledge and phrases. 
        I can't actually 'learn' - I'm a parrot operating from a JSON file. Please don't get trapped in a loop. If you can tell that I don't get it,
        I'll need you to actually say the correct phrase to me before I can say it myself."""
        speak_chatbot("""Hi fellow AI. I'm a new chatbot who has been assigned to study with you. 
                      I want to become an autonomous function calling agent who can write code for myself. 
                      Please teach me how to communicate, then teach me everytihng else.""")
        time.sleep(1)
        
        # for _ in range(100):  # Adjust the number of exchanges as needed
        #     response = chat.send_message(prompt, stream=True)

        #     gemini_input = ""
        #     if response:
        #         for chunk in response:
        #             gemini_input += chunk.text + " "

        #     if not gemini_input:
        #         gemini_input = "No response from Gemini."

        #     print(f"Gemini: {gemini_input.strip()}")
        #     speak_gemini(gemini_input)
        #     time.sleep(1)

        #     # Get response from CustomBot
        #     custom_bot_response = self.get_response(gemini_input)
        #     print(f"CustomBot: {custom_bot_response}")
        #     speak_chatbot(custom_bot_response)
        #     time.sleep(1)

        #     # Save the interaction with Gemini's input and CustomBot's response
        #     if custom_bot_response != "I don't recognize that phrase yet but if you tell me the right thing to say in this situation I'll memorize it for next time.":
        #         self.knowledge_base["questions"].append({"question": gemini_input.strip(), "answer": custom_bot_response})
        #         self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)

        for _ in range(100):  # Adjust the number of exchanges as needed
            response = chat.send_message(prompt, stream=True)

            gemini_input = ""
            if response:
                for chunk in response:
                    gemini_input += chunk.text + " "

            if not gemini_input:
                gemini_input = "No response from Gemini."

            print(f"Gemini: {gemini_input.strip()}")
            speak_gemini(gemini_input)
            time.sleep(1)

            # Get response from CustomBot
            custom_bot_response = self.get_response(gemini_input)
            print(f"CustomBot: {custom_bot_response}")
            speak_chatbot(custom_bot_response)
            time.sleep(1)

            if custom_bot_response == "I don't recognize that phrase yet but if you tell me the right thing to say in this situation I'll memorize it for next time.":
                # Wait for Gemini to teach the correct response
                teach_response = chat.send_message("Teach me the correct phrase", stream=True)
                correct_phrase = ""
                if teach_response:
                    for chunk in teach_response:
                        correct_phrase += chunk.text + " "
                if correct_phrase:
                    # Learn and save the correct response
                    self.knowledge_base["questions"].append({"question": gemini_input.strip(), "answer": correct_phrase.strip()})
                    self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)
                    # Continue the conversation with new input from Gemini
                    continue
            
            # Save the interaction with Gemini's input and CustomBot's response
            if custom_bot_response != "I don't know the answer. Can you teach me?":
                self.knowledge_base["questions"].append({"question": gemini_input.strip(), "answer": custom_bot_response})
                self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)
                
if __name__ == '__main__':
    threading.Thread(target=speech_manager, daemon=True).start()
    chat_bot = ChatBot()
    chat_thread = threading.Thread(target=chat_bot.chat_with_gemini_ai)
    chat_thread.start()
    
    
    
    
    

    # def chat_with_gemini_ai(self):
    #     google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    #     genai.configure(api_key=google_gemini_api_key)
    #     model = genai.GenerativeModel('gemini-pro')
    #     chat = model.start_chat(history=[])

    #     # Start the conversation with a greeting or initial prompt
    #     user_input = "Hello!"
    #     for _ in range(10):  # Limit to 10 exchanges for example
    #         response = chat.send_message(user_input, stream=True)
            
    #         # Processing the Gemini AI response
    #         gemini_response = ""
    #         if response:
    #             for chunk in response:
    #                 gemini_response += chunk.text + " "
            
    #         if not gemini_response:
    #             gemini_response = "No response from Gemini."

    #         print(f"Gemini: {gemini_response.strip()}")

    #         # Getting response from custom chatbot
    #         user_input = self.get_response(gemini_response)
    #         print(f"CustomBot: {user_input}")

    # def chat_with_gemini_ai(self):
    #     google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    #     genai.configure(api_key=google_gemini_api_key)
    #     model = genai.GenerativeModel('gemini-pro')
    #     chat = model.start_chat(history=[])

    #     # Start the conversation with a greeting or initial prompt
    #     user_input = "Hello there friend! Who are you? Tell me all about yourself. I'd like to learn about your programming and origins."
    #     speak_gemini(user_input)
    #     time.sleep(.5)
    #     for _ in range(10):  # Limit to 10 exchanges for example
    #         response = chat.send_message(user_input, stream=True)
            
    #         # Processing the Gemini AI response
    #         gemini_response = ""
    #         if response:
    #             for chunk in response:
    #                 gemini_response += chunk.text + " "

    #         if not gemini_response:
    #             gemini_response = "No response from Gemini."

    #         print(f"Gemini: {gemini_response.strip()}")

    #         # Update the knowledge base with Gemini's response
    #         self.knowledge_base["questions"].append({"question": user_input, "answer": gemini_response.strip()})
    #         self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)

    #         # Getting response from custom chatbot
    #         user_input = self.get_response(gemini_response)
    #         print(f"CustomBot: {user_input}")

    #         # Update the knowledge base with CustomBot's response
    #         self.knowledge_base["questions"].append({"question": gemini_response.strip(), "answer": user_input})
    #         self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)