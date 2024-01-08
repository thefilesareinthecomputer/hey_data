'''command line interface for text generation with gemini - can be improved as a loop'''

from dotenv import load_dotenv
import certifi
import google.generativeai as genai
import os
import ssl

# bring in the environment variables from the .env file
load_dotenv()

google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

genai.configure(api_key=google_gemini_api_key)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

response = chat.send_message("what data does google store from users or developers when using the speech recognition (aka speech_recognition) python package? are there any comparable open source alternatives? are there any libraries that do this and don't store users' data? can google users or devs opt out of data storage with the speech recognition api?", stream=True)

for chunk in response:
    print(chunk.text)