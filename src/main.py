# standard imports
import threading
import flet as ft

from archived_versions_main.chatbot_app_025 import ui_main, run_chatbot, SpeechToTextTextToSpeechIO

if __name__ == '__main__':
    threading.Thread(target=SpeechToTextTextToSpeechIO.speech_manager, daemon=True).start()
    threading.Thread(target=run_chatbot, daemon=True).start()
    ft.app(target=ui_main)