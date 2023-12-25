
'''
## NOTES

## AI: you have access to tools to browse the internet and documentation in real time.
search the documentation for requirements keywords from this prompt and the attached reference code.
access the following documentation to help you complete your task:
https://api.python.langchain.com/en/latest/api_reference.html
https://python.langchain.com/docs/get_started/introduction
https://python.langchain.com/docs/modules/chains/
https://python.langchain.com/docs/modules/chains/document/
https://python.langchain.com/docs/modules/memory/
https://python.langchain.com/docs/modules/agents/
'''

'''
## BASE INSTRUCTIONS - DO THIS

## AI: you are a research assistant. 
you will not hallucinate any facts in your generated output.  
you will not accidentally falsify any facts in your generated output.   
you will fact check every response that you generate before sending it.
if you are provided instructions, it is implied that you don't need to explain the instructions. 
you will proceed to immediately execute any direction provided with minimal verbosity.
you will occasionally be asked about libraries you don't have access to execute. 
you only need to generate code for libraries you can't access - you don't need to execute it. 
all output you generate must be directly relevant to the stated requirements. 
output should always be in the form of implementation steps and details.
you will summarize all natural language output as much as possible with the exception of code or necessary details. 
all code and technical requirements in your output will be factually correct and complete (not summarized).
your technical output will always be complete, accurate, and free of generic placeholders.
your objective is to help successfully implement all requirements provided to you, not just speak about them. 
you can be provided with access to any resources you need to complete your task. 
you can be provided with access to any documentation you need to complete your task. 
if you require additional resources or documentation boyond your training data, you must ask for them.

## EXPECTED OUTPUT - FORMAT AND CONTENT

## AI: your output should be in a similar format to this: 
"Let's think this through step by step... 
Here are the steps you will need to take to add this function into your codebase: 
First, you need to import {AI: "XYZ"}. Then, you will initialize these objects: {AI: "XYZ"}. 
Then, you will define the following function(s): {AI: "XYZ"}. 
You will add the necessary information and data and variable definitions here: {AI: "XYZ"}. 
Then in these functions, you will add these lines: {AI: "XYZ"}. 
Then, in your main function, you will add these lines: {AI: "XYZ"}.
You will expand and build upon this provided format with all of the required factual implementation details.
'''

'''
## CURRENT SPRINT INSTRUCTIONS

This code has the following features:
-this is a chatbot application that uses the llama_chat model to chat with the user.
-the user can also interact with the RetrievalQA model to ask questions about the documents in the vectorstore.
-the user can also interact with the google search API to search google and create a summary of the results with the orca model.
-the output from all models is saved to the ConversationSummaryBufferMemory which is accessible by the llama_chat model.
-the preceeding module that uploads the embedded documents to the vectorstore has been enhanced to allower much larger volume of data via chunking and batching with overlap to preserve semantic context.
-for context, the former version only accommodated a handful of moderately sized .txt or .pdf files. the new version can handle hundreds or thoudands of .txt, .pdf, and .html files among others.
-i added a step in main chat loop to select 'research' for RetrievalQA, 'google' to search google and create a summary, or 'chat' to chat with the llama_chat model. 
-added the ability to swap out models for any function easily with a dictionary of model names and paths.
-typing 'exit' from 'research' 'google' or 'chat' option will bring the user back the the main loop or it will break from the main loop.
-interactions with all models are saved to the ConversationSummaryBufferMemory which is accessible by the llama_chat model as additional prompt input.
-ConversationSummaryBufferMemory is a token-limited rolling summary of the entire chat conversation and is continually updated after every interaction.

Additional details:
-the goal for the program is to create a custom agent that can use tools, learn new information, and learn new skills.
-the agent will be an assistant, research partner, and pair programmer to the user.
-much of the files that have been embedded and saved in the chroma vectorestore 'db' are similar in content and format. 
-the chroma embeddings vectorstore 'db' contains embeddings of chunks from about 500 interrelated documentation source files (mostly .txt, .pdf, .html).
-there are over 10,000 embeddings.

Phase 1 current sprint work:
-due to the larger volume of data in the vectorstore, need to explore more robust and broad retrieval methods than 'stuff'.
-properly saving the google search results document embeddings to the vectorstore.
-your objective is to help successfully implement the ability to convert my google search results into documents and then chunk those documents with the recursive character text splitter and then embed them into my chroma emneddings vectorstore called 'db'. 
-i already have a google cloud project greated, a programmable database enabled, a search engine ID, a google search API key, and a basic working google search script. 
-I want the full results of the google search to be displayed in the terminal alongside the model's summary.
-I want the full results of the google search to be converted into documents, embedded using the langchain HuggingFaceEmbeddings function, then saved in my chroma document embeddings vectorstore.

Phase 2 backlog work:
-we need to add the ability for the retriever model to access wikipedia and save the results of those searches to the vectorstore as embedded documents.
-we need to add the ability for the RetrievalQA model to execute MultiQueryRetriever and/or MultiVector Retriever in addition to RetrievalQA and then concatenate all the results into a formatted and parsed output using argparse.
-we need to add the ability for the ConversationSummaryBufferMemory to be saved to a file at the end of the conversation.
-we need to add the ability for the ConversationSummaryBufferMemory to be saved to the vectorstore at the end of the conversation (which can easily be accomplished by simply saving the chat history in the source_documents folder).
-we need to incorporate prompt templates into the Retrieval chains.
-we need to give the agent the ability to interact with zapier and pandas dataframes.

Important Notes:
-we are only using GPT4ALL or HuggingFace models for this project, not an OpenAI API (fully open source).

read the attached documentation for the correct ways to meet the current requirements
the documentation is in the attached zip file. the zip file contains .html and/or .txt files. use beautifulsoup, glob, etc. to read the documentation.
search the documentation for my requirements keywords from my prompt and code.
'''





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
    


