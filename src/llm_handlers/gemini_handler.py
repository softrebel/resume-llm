from . import BaseLLMHandler
import google.generativeai as genai
from google.generativeai.generative_models import ChatSession
from src._core import project_configs


class GeminiHandler(BaseLLMHandler):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str | None = None,
        chat_history: list[dict] = [],
        generation_config: dict | None = None,
    ):
        super().__init__()
        if not api_key:
            api_key = project_configs.GEMINI_API_KEY
        genai.configure(api_key=api_key)
        self.__model = genai.GenerativeModel(
            model_name, generation_config=generation_config
        )
        self.__chat: ChatSession | None = None
        self.__chat_history: list[dict] = chat_history

    @property
    def chat(self):
        if self.__chat is None:
            self.__chat = self.__model.start_chat(history=self.__chat_history)
        return self.__chat

    @chat.setter
    def set_chat(self, chat: ChatSession):
        self.__chat = chat

    def send_prompt(self, prompt: str):
        response = self.chat.send_message(prompt)
        return response.text
