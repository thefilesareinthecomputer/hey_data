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

response = chat.send_message("Please give me instructions for how to pass a downloaded image to gemini provision in a generate content message. write your output in python code please.", stream=True)

for chunk in response:
    print(chunk.text)