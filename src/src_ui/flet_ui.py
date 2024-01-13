import flet
from flet import Page, ElevatedButton, UserControl, Row

class ChatbotUI(UserControl):
    def build(self):
        self.mic_button = ElevatedButton(
            text="Hold to Talk",
            on_down=self.on_mic_button_down,
            on_up=self.on_mic_button_up
        )

        return Row([self.mic_button])

    def on_mic_button_down(self, e):
        # Start microphone listening
        print("Mic on")  # Replace with your chatbot's start listening method

    def on_mic_button_up(self, e):
        # Stop microphone and process command
        print("Mic off")  # Replace with your chatbot's stop listening method

def main(page: Page):
    page.title = "Chatbot UI"
    page.add(ChatbotUI())

flet.app(target=main)