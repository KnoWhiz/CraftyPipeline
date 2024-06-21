import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv  # pip install python-dotenv
import openai
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI

class LLMApiHandler(ABC):
    """
    Abstract base class for LLM API Handlers.
    """

    @abstractmethod
    def load_model(self, temperature, model_name):
        """
        Load and return the model.
        """
        pass

class OpenAiHandler(LLMApiHandler):
    def __init__(self, api_key_dir, organization='your-organization-id'):
        load_dotenv(api_key_dir)
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.organization = organization

    def load_model(self, temperature, model_name):
        try:
            # model = ChatOpenAI(temperature=temperature, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name=model_name)
            model = ChatOpenAI(temperature=temperature, streaming=False, callbacks=[BaseCallbackHandler()], model_name=model_name, verbose=False)
            # model = ChatOpenAI(temperature=temperature, streaming=False, callbacks=[ConsoleCallbackHandler()], model_name=model_name, verbose=False)
            # print(f'Successfully loaded {model_name}!')
            return model
        except Exception as e:
            print(f'Failed to load {model_name} due to {e}')
            return None

class LLMApiFactory:
    """
    Factory class to create LLM API handler instances.
    """
    @staticmethod
    def get_api_handler(para):
        # print("\nLoading LLM API handler...")
        llm_source = para['llm_source'].lower()
        if llm_source == 'openai':
            return OpenAiHandler(".env")
        # Extend here with elif statements for other LLM sources
        else:
            raise ValueError(f'LLM source {llm_source} is not supported.')

class ApiHandler:
    def __init__(self, para):
        self.api_handler = LLMApiFactory.get_api_handler(para)
        self.models = self.load_models(para)

    def load_models(self, para):
        models = {
            'basic': {
                'instance': self.api_handler.load_model(para['temperature'], 'gpt-3.5-turbo-0125'),
                'context_window': 16385
            },
            'advance': {
                'instance': self.api_handler.load_model(para['temperature'], 'gpt-4o'),
                'context_window': 128000
            },
            'creative': {
                'instance': self.api_handler.load_model(para['creative_temperature'], 'gpt-4o'),
                'context_window': 128000
            },
        }
        return models
