'''this is a rudimentary chatbot that will state whenever it doesn't recognize a statement, then prompt the usewr to tell it the correct response, 
then will learn to immediately parrot that response. there is flexibility for a % match for input, but the output is 100% hard coded.'''

import json
import os
from difflib import get_close_matches
from dotenv import load_dotenv
import google.generativeai as genai

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

    def run(self):
        while True:
            user_input = input('You: ')
            
            if user_input.lower() == 'quit':
                break
            
            best_match = self.find_best_match(user_input, [q["question"] for q in self.knowledge_base["questions"]])
            
            if best_match:
                answer = self.get_answer_for_question(best_match)
                print(f'Bot: {answer}')
            else:
                print('Bot: I don\'t know the answer. Can you teach me?')
                new_answer = input('Type the answer or "skip" to skip: ')
                
                if new_answer.lower() != 'skip':
                    self.knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                    self.save_knowledge_base(f'{self.base_knowledge_dir_path}/knowledge_base.json', self.knowledge_base)
                    print('Bot: Thank you! I learned a new response!')

    # @staticmethod
    # def chat_with_ai_agent_user():
    #     google_gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    #     genai.configure(api_key=google_gemini_api_key)
    #     model = genai.GenerativeModel('gemini-pro')
    #     chat = model.start_chat(history=[])
    #     while True:
    #         user_input = input('You: ')
    #         response = chat.send_message(f'{user_input}')
    #         if response:
    #             response.resolve()
    #             ai_response = response.text
    #             print(ai_response)
                
if __name__ == '__main__':
    chat_bot = ChatBot()
    # ChatBot.chat_with_ai_agent_user()
    chat_bot.run()
    # Use chat_bot.run_agent() for chats with Gemini

