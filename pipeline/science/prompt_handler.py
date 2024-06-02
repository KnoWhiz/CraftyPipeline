import tiktoken  # Assuming tiktoken library is available for token encoding
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pipeline.science.api_handler import ApiHandler

class PromptHandler:
    def __init__(self, api_handler):
        """
        Initializes the PromptHandler with a reference to an ApiHandler instance.
        """
        self.api_handler = api_handler

    def get_model_context_window(self, model_name):
        """
        Retrieves the context window size for a given model from the ApiHandler instance.
        """
        model_info = self.api_handler.models[model_name]
        if model_info:
            return model_info['context_window']
        else:
            raise ValueError(f"Model {model_name} not found in ApiHandler.")

    def get_tokens_number_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """
        Calculates the number of tokens in a string using a specific encoding.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def split_prompt(self, input_text, model_name, encoding_name="cl100k_base", custom_token_limit=None, return_first_chunk_only=False):
        """
        Prepares the input prompt to fit within a specified token limit by dividing it into chunks
        based on the actual token count, using a binary search approach to improve efficiency.
        Allows for a custom token limit overriding the model's default. Optionally returns just the
        first chunk if specified.

        Parameters:
        - input_text (str): The text to be processed.
        - model_name (str): The name of the model, which determines the default context window size.
        - encoding_name (str, optional): The name of the encoding to use for tokenization.
        - custom_token_limit (int, optional): A custom token limit for each chunk, overriding the model's default.
        - return_first_chunk_only (bool, optional): If set to True, the method returns after processing the first chunk.

        Returns:
        - list[str]: A list of text chunks, each fitting within the specified token limit. If `return_first_chunk_only`
        is True, returns a list containing only the first chunk.
        """
        token_limit = custom_token_limit if custom_token_limit is not None else self.get_model_context_window(model_name)
        print(f'token_limit: {token_limit}')
        chunks = []
        start_index = 0

        while start_index < len(input_text):
            low, high = start_index, len(input_text)

            # Use binary search to find the largest possible chunk
            while low < high:
                mid = (low + high + 1) // 2
                substring = input_text[start_index:mid]
                token_count = self.get_tokens_number_from_string(substring, encoding_name)

                if token_count <= token_limit:
                    low = mid  # Try a larger size
                else:
                    high = mid - 1  # Try a smaller size

            # After finding the largest fitting substring, add it to chunks
            chunk = input_text[start_index:low]
            if not chunk:  # This should not happen, but adds safety
                raise ValueError("Unable to process a segment of the text within the token limit.")
            chunks.append(chunk)
            start_index = low  # Move start index past this chunk

            if return_first_chunk_only:
                break  # Stop processing and return after the first chunk if specified

        return chunks

    def summarize_prompt(self, input_text, model_name, encoding_name="cl100k_base", custom_token_limit=None, return_first_chunk_only=False):
        chunks = self.split_prompt(input_text, model_name, encoding_name=encoding_name, custom_token_limit=custom_token_limit, return_first_chunk_only=return_first_chunk_only)
        n = len(chunks)
        max_token = int(custom_token_limit/n)
        self.llm_basic = self.api_handler.models[model_name]['instance']
        if n == 1:
            qdocs_supps_summary = input_text
        else:
            qdocs_supps_summary = ""
            for i in range(len(chunks)):
                parser = StrOutputParser()
                prompt = ChatPromptTemplate.from_template(
                        """
                        please summarize the following text in {max_token} tokens or less.
                        "\n\n text: {text}"
                        """)
                chain = prompt | self.llm_basic | parser
                response = chain.invoke({'text': chunks[i], 'max_token': max_token})
                qdocs_supps_summary = " ".join([qdocs_supps_summary, response])
        return qdocs_supps_summary