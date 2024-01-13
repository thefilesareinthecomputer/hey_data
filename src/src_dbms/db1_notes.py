from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
import json
from datetime import datetime

load_dotenv()
JAVA_HOME = os.getenv('JAVA_HOME')
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_PATH = os.getenv("NEO4J_PATH")
USER_DOWNLOADS_FOLDER = os.getenv('USER_DOWNLOADS_FOLDER')
PROJECT_VENV_DIRECTORY = os.getenv('PROJECT_VENV_DIRECTORY')
PROJECT_ROOT_DIRECTORY = os.getenv('PROJECT_ROOT_DIRECTORY')
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATABASES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_databases')
FILE_DROP_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_generated_files')
LOCAL_LLMS_DIR = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_local_models')
BASE_KNOWLEDGE_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_base_knowledge')
SECRETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_secrets')
SOURCE_DATA_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'app_source_data')
SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'src')
TESTS_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, '_tests')
UTILITIES_DIR_PATH = os.path.join(PROJECT_ROOT_DIRECTORY, 'utilities')
folders_to_create = [DATABASES_DIR_PATH, FILE_DROP_DIR_PATH, LOCAL_LLMS_DIR, BASE_KNOWLEDGE_DIR_PATH, SECRETS_DIR_PATH, SOURCE_DATA_DIR_PATH, SRC_DIR_PATH, TESTS_DIR_PATH, UTILITIES_DIR_PATH]
for folder in folders_to_create:
    if not os.path.exists(folder):
        os.makedirs(folder)
    

def save_note(subject, text):
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with driver.session() as session:
            session.run("""
                CREATE (n:Note {subject: $subject, text: $text, timestamp: $timestamp})
            """, subject=subject, text=text, timestamp=timestamp)
        speak_mainframe('Saved.')
    except Exception as e:
        speak_mainframe("Error occurred. Please try again.")
        print(e)
    finally:
        driver.close()
    
def recall_notes(subject):
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("""
                MATCH (n:Note {subject: $subject})
                RETURN n.text, n.timestamp
                ORDER BY n.timestamp DESC
            """, subject=subject)
            notes = [record['n.text'] for record in result]
        if notes:
            speak_mainframe(" ".join(notes))
        else:
            speak_mainframe("No notes found for this subject.")
    except Exception as e:      
        speak_mainframe("Error occurred. Please try again.")
        print(e)
    finally:
        driver.close()