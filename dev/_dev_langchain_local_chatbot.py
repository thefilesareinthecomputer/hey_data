'''under development'''

import argparse
import chromadb
import csv
import gc
import json
import os
import streamlit as st
import time
from contextlib import contextmanager
from dotenv import load_dotenv
# from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.chains import RetrievalQA, ConversationChain, SequentialChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Directories and relative file paths
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_VENV_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_VENV_DIR)
PROJECT_MODELS_DIR = os.path.join(PROJECT_VENV_DIR, "local_models")
CHAT_MODEL_PATH = os.path.join(PROJECT_MODELS_DIR, 'mistral-7b-openorca.Q4_0.gguf')

embeddings_model_name = 'all-MiniLM-L6-v2'
persist_directory = 'db'
model_type = 'GPT4All'
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

print(SCRIPT_DIR)
print(PROJECT_VENV_DIR)
print(PROJECT_ROOT_DIR)
print(PROJECT_MODELS_DIR)
print(CHAT_MODEL_PATH)
print()

chat_model = GPT4All(model=CHAT_MODEL_PATH, verbose=True)
conversation_summary_buffer_memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=100)
conversation_with_summary = ConversationChain(llm=chat_model, memory=conversation_summary_buffer_memory, verbose=True)

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')

#     parser.add_argument("--mute-stream", "-M",
#                         action='store_true',
#                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

#     return parser.parse_args()


# args = parse_arguments()  # Parse the command line arguments


def chat():
    chat_input = input("\nYou: ")
    
    # Chat loop
    while True:
        if chat_input.lower() == "exit":
            break  
        # Get model's response
        response = conversation_with_summary.predict(input=chat_input)

        # Save the context to the memory
        conversation_summary_buffer_memory.save_context({"input": chat_input}, {"output": response})

        # Display the response
        print("\nAI Response:", response)
        print("\n-------------------------")
        
        # Prompt the user for the next input within the chat loop
        chat_input = input("\nContinue chatting or say 'exit' to return to the main menu: ")  


if __name__ == "__main__":
    chat()
    


