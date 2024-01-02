# IMPORTS ###################################################################################################################################

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from io import StringIO
from math import radians, cos, sin, asin, sqrt
from openai import OpenAI
from pyppeteer import launch # new
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from urllib.parse import urlparse, urljoin
import asyncio # new
import calendar
import certifi
import datetime
import google.generativeai as genai
import json
import numpy as np
import os
import pandas as pd
import pickle
import PIL.Image
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
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
ARCHIVED_DEV_VERSIONS_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_archive')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
NOTES_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')

folders_to_create = [ARCHIVED_DEV_VERSIONS_PATH, FILE_DROP_DIR_PATH, LOCAL_LLMS_DIR, NOTES_DROP_DIR_PATH, SECRETS_DIR_PATH, SOURCE_DATA_DIR_PATH, SRC_DIR_PATH, TESTS_DIR_PATH, UTILITIES_DIR_PATH]
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

# CLASSES ###################################################################################################################################
    
class Event:
    def __init__(self, summary, start_time, end_time, description='', location=None, attendees=None, recurrence=None, event_id=None):
        self.summary = summary
        self.start_time = start_time
        self.end_time = end_time
        self.description = description
        self.location = location
        self.attendees = attendees if attendees else []
        self.recurrence = recurrence
        self.event_id = event_id  # Google Calendar Event ID, if available

    def to_google_calendar_format(self):
        event = {
            'summary': self.summary,
            'description': self.description,
            'start': {'dateTime': self.start_time.isoformat(), 'timeZone': 'UTC'}, 
            'end': {'dateTime': self.end_time.isoformat(), 'timeZone': 'UTC'},
        }

        if self.location:
            event['location'] = self.location

        if self.attendees:
            event['attendees'] = [{'email': email} for email in self.attendees]

        if self.recurrence:
            event['recurrence'] = [self.recurrence]

        return event

    @staticmethod
    def from_google_calendar_data(google_event_data):
        return Event(
            summary=google_event_data.get('summary'),
            start_time=datetime.fromisoformat(google_event_data['start']['dateTime']),
            end_time=datetime.fromisoformat(google_event_data['end']['dateTime']),
            description=google_event_data.get('description', ''),
            location=google_event_data.get('location'),
            attendees=[attendee['email'] for attendee in google_event_data.get('attendees', [])],
            recurrence=google_event_data.get('recurrence'),
            event_id=google_event_data.get('id')
        )

class Reminders:
    def __init__(self, service):
        self.service = service

    def add_event(self, event):
        google_event = event.to_google_calendar_format()
        created_event = self.service.events().insert(calendarId='primary', body=google_event).execute()
        print(f"Event created: {created_event.get('htmlLink')}")
        return Event.from_google_calendar_data(created_event)

    def get_event(self, event_id):
        google_event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
        return Event.from_google_calendar_data(google_event)
    
# FUNCTIONS ###################################################################################################################################

def authenticate_google_api():
    scopes = ['https://www.googleapis.com/auth/calendar']
    creds = None

    # Token file to store user's access and refresh tokens
    token_path = f'{SECRETS_DIR_PATH}/token.pickle'

    # Load credentials if they exist
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(f'{SECRETS_DIR_PATH}/credentials.json', scopes)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return creds

# Initialize Google Calendar API service
creds = authenticate_google_api()
service = build('calendar', 'v3', credentials=creds)

# MAIN EXECUTION ###################################################################################################################################
