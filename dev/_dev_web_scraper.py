
# IMPORTS ###################################################################################################################################

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from dotenv import load_dotenv
from fake_useragent import UserAgent
from math import radians, cos, sin, asin, sqrt
from openai import OpenAI
from pyppeteer import launch, errors as pyppeteer_errors
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from urllib.parse import urlparse, urljoin
from webdriver_manager.chrome import ChromeDriverManager
import asyncio # new
import certifi
import datetime
import difflib
import google.generativeai as genai
import json
import logging
import numpy as np
import os
import pandas as pd
import pyautogui
import pytz
import queue
import random
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
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
USER_SELECTED_HOME_LAT = os.getenv('USER_SELECTED_HOME_LAT', 'None')  # Float with 6 decimal places
USER_SELECTED_HOME_LON = os.getenv('USER_SELECTED_HOME_LON', 'None')  # Float with 6 decimal places 
USER_SELECTED_TIMEZONE = os.getenv('USER_SELECTED_TIMEZONE', 'America/Chicago')  # Country/State format
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

# Set API keys and other information from environment variables
open_weather_api_key = os.getenv('OPEN_WEATHER_API_KEY')
wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
openai_api_key=os.getenv('OPENAI_API_KEY')
google_cloud_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

# Initialize LLM
genai.configure(api_key=google_gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

# WEB SCRAPING CLASSES AND FUNCTIONS ###################################################################################################################################
        
class SeleniumWebScraper:
    def __init__(self):
        self.set_random_user_agent()
        self.scraped_data = []  # Initialize scraped_data here
        
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Linux; Android 10; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 Mobile Safari/537.36',
        'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/604.1 (KHTML, like Gecko) Edge/18.19582',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36',
        'Mozilla/5.0 (iPad; CPU OS 13_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_6; en-us) AppleWebKit/533.20.25 (KHTML, like Gecko) Version/5.0.4 Safari/533.20.27',
        'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19577',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/601.1 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/601.1',
        'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 12_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.8 (KHTML, like Gecko) Version/9.1.2 Safari/601.7.7',
        'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0',
    ]
    
    def set_random_user_agent(self):
        user_agent = random.choice(self.USER_AGENTS)
        options = webdriver.ChromeOptions()
        options.add_argument(f'user-agent={user_agent}')

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        except WebDriverException as e:
            print(f"Error initializing ChromeDriver: {e}")
            raise e
            # BACKUP DRIVER HERE

    def find_menu_link(self):
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        links = soup.find_all('a')
        menu_keywords = ['dinner menu', 'food menu', 'lunch menu', 'food', 'menu']

        for link in links:
            link_text = link.get_text().lower()
            link_href = link.get('href', '').lower()
            if any(keyword in link_text or keyword in link_href for keyword in menu_keywords):
                return link_href

        return None

    def scrape_website_data(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            menu_url = self.find_menu_link()
            if menu_url:
                self.take_screenshot_of_menu(menu_url)
            else:
                print("No menu link found on the page.")

            if self.scraped_data:
                self.save_data_as_json()
            else:
                print("No data scraped.")

        except Exception as e:
            print(f"Error during website scraping: {e}")
        finally:
            self.close()

    def take_screenshot_of_menu(self, menu_url):
        try:
            # If the menu URL is relative, make it absolute
            if not menu_url.startswith('http'):
                menu_url = os.path.join(URL, menu_url)

            self.driver.get(menu_url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            screenshot_file = f'{FILE_DROP_DIR_PATH}/menu_page_screenshot.png'
            self.driver.save_screenshot(screenshot_file)
            self.scraped_data.append(screenshot_file)

        except Exception as e:
            print(f"Error taking screenshot of menu page {menu_url}: {e}")

    def save_data_as_json(self):
        filename = os.path.join(os.getenv('FILE_DROP_DIR_PATH', '.'), 'scraped_data.json')
        screenshot_info = [{'screenshot_file': screenshot} for screenshot in self.scraped_data]
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(screenshot_info, file, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")

    def close(self):
        if self.driver:
            self.driver.quit()

# MAIN CODE EXECUTION ###################################################################################################################################
URL = os.getenv('URL')
scraper = SeleniumWebScraper()
scraper.scrape_website_data(URL)
















































# class AgentWebScraper:
#     def __init__(self, model):
#         self.model = model
#         '''
#         menu tab navigation logic notes
#         see the example dictionary with base links and menu links above for guidance on the types of menu links to look for
#         there are many possible variaitons of link names, button names, and click paths to these menus
#         the first preference should be 'dinner menu', then if not present, 'food menu', then 'dinner', then 'food', then 'menu', etc.
#         this will ensure thet on pages with multiple menus we can navigate to the most like menu to be the dinner menu first
#         note that the actual menu itself will likely be 1 click deeper than the menu link because most of them are nested or embedded or pdf links
#         we need to gather the actual content of the menu, not just the link to the menu
#         we need to make sure the scraping logic can handle the different types of menus and menu links and the various navigation paths to the menu
#         we should recursively navigate the site and the scraper should move in reverse if it encounters a barrier and the navigation logic that is seeking dinner menus should be enacted at each forward moving interval
#         the scraper must be able to x out of pop up windows and defer to the user for capthcas and other barriers that it can't handle itself
#         any failure should just send the scraper to the next link in the list
#         the scraping sequence for each website should not end until it obtains data or fails to obtain data from each link in the list
#         '''

#     menu_link_examples = {
#         'https://rh.com/us/en/marin/restaurant': 'https://images.restorationhardware.com/media/mobile-menus/VCM_Dinner_Mobile.pdf',
#         'https://www.perkinsrestaurants.com/': 'https://www.perkinsrestaurants.com/menu',
#         'https://hillstonerestaurant.com/locations/phoenix/': 'https://hillstone.com/menus/hillstone/Hillstone%20Phoenix%20Food.pdf?version=v-1703788018',
#         'https://www.houstons.com/locations/scottsdale/?_gl=1*fns6p4*_ga*MTMyOTg5MDg3LjE3MDE4MjMzNTQ.*_ga_SNRLFE0YFZ*MTcwMzc4ODAxMS41LjEuMTcwMzc4OTY3MS4wLjAuMA..': 'https://hillstone.com/menus/houstons/Houstons%20Scottsdale%20Food.pdf?version=v-1703788044',
#         'https://banderarestaurants.com/?_gl=1*plwdgy*_ga*MTMyOTg5MDg3LjE3MDE4MjMzNTQ.*_ga_SNRLFE0YFZ*MTcwMzc4ODAxMS41LjEuMTcwMzc4OTY3MS4wLjAuMA..': 'https://hillstone.com/menus/bandera/Bandera%20Corona%20del%20Mar%20Dinner.pdf?version=v-1703788173',
#         'https://honorbar.com/locations/beverlyhills/?_gl=1*plwdgy*_ga*MTMyOTg5MDg3LjE3MDE4MjMzNTQ.*_ga_SNRLFE0YFZ*MTcwMzc4ODAxMS41LjEuMTcwMzc4OTY3MS4wLjAuMA..': 'https://hillstone.com/menus/honorbar/Honor%20Bar%20Beverly%20Hills%20Menu.pdf?version=v-1703788276',
#         'https://gulfstreamrestaurant.com/?_gl=1*1ulivrw*_ga*MTMyOTg5MDg3LjE3MDE4MjMzNTQ.*_ga_SNRLFE0YFZ*MTcwMzc4ODAxMS41LjEuMTcwMzc4OTY3MS4wLjAuMA..': 'https://hillstone.com/menus/gulfstream/Gulfstream%20Food.pdf?version=v-1703788301',
#         'https://losaltosgrill.com/': 'https://hillstone.com/menus/losaltosgrill/Los%20Altos%20Grill%20Dinner.pdf?version=v-1703788814',
#         'https://www.handrollbar.com/locations/nomad/': 'https://www.handrollbar.com/new-york-menu/',
#         'https://www.boccadibacconyc.com/': 'https://www.boccadibacconyc.com/menus-chelsea/#dinner',
#         'https://www.chrnyc.com/': 'https://www.chrnyc.com/menus#dinner',
#         'https://lezie.myshopify.com/': 'https://lezie.myshopify.com/pages/dinner',
#         'https://westvillenyc.com/': 'https://westvillenyc.com/menu/dinner/',
#         'https://www.fourseasons.com/sanfrancisco/dining/restaurants/mkt_restaurant_bar/': 'https://www.fourseasons.com/sanfrancisco/dining/restaurants/mkt_restaurant_bar/dinner/',
#         'https://www.one65sf.com/': 'https://one65sf.com/menus/o/2021/nyemenu.pdf',
#         'https://www.properhotel.com/': 'https://www.properhotel.com/wp-content/uploads/2023/12/21082325/Villon-Dinner-11.19.23.pdf',
#         'https://frenchsoulfood.com/': 'https://frenchsoulfood.com/creole-comfort-food/',
#     }
    
#     def format_links_and_ask_llm(self, links):
#         link_options = {str(index): link.get_attribute('href') for index, link in enumerate(links)}
#         print("Formatted link options for LLM:", link_options)  # Print formatted links
#         return self.ask_llm_a_question(self.menu_link_examples, link_options)

#     def ask_llm_a_question(self, menu_link_examples, link_options):
#         try:
#             prompt = (
#                 "Hi Gemini, I need your help to determine the most likely link to the dinner menu from a list of links on a restaurant's website. "
#                 "Here are some examples of website base URLs and their corresponding dinner menu links for reference:\n" +
#                 "\n".join([f"{base_url}: {menu_url}" for base_url, menu_url in menu_link_examples.items()]) +
#                 "\n\nNow, I have a list of links from a different website. Please analyze these links and tell me which one is most likely the dinner menu. "
#                 "The links are as follows:\n" +
#                 "\n".join([f"{key}: {url}" for key, url in link_options.items()]) +
#                 "\nWhich link number do you think leads to the dinner menu?"
#             )
#             print("Asking LLM:", prompt)  # Print the LLM prompt
#             response = self.model.generate_content(prompt)
#             summary = response.text
#             # Use regex to find the first sequence of digits in the response
#             match = re.search(r'\d+', summary)
#             if match:
#                 best_link_key = match.group()
#                 print("LLM selected link:", best_link_key)  # Print the selected link
#             else:
#                 print("No valid link key found in LLM response.")
#                 best_link_key = None

#             return best_link_key
#         except Exception as e:
#             print(f"Error asking LLM a question: {e}")
#             return None

