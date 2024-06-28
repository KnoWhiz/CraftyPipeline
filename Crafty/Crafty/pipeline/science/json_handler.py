from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json

class JsonHandler:
    def __init__(self, llm):
        """
        Initializes the PromptHandler with a reference to an ApiHandler instance.
        """
        self.llm = llm

    def fix_json_formatting(self, json_string):
    # Fix common JSON formatting issues here
        print(json_string)
        parser1 = JsonOutputParser()
        prompt1 = ChatPromptTemplate.from_template(
                """
                    "Generate a well-formatted JSON file by resolving any formatting errors present in the provided JSON string:"
                    "\n\n JSON string: {json_string}."
                    "\n\n The given JSON string contains multiple nested layers."
                    Return pure json file.
                    Do not include any additional information in the output.
                    Do not include any \"```\" in the output.
                """
                )
        chain1 = prompt1 | self.llm | parser1
        response = chain1({'json_string': json_string})
        print("\nCorrected json format: ", response)
        return response

    def load_json_with_fixes(self, json_string):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}. Attempting to fix formatting...")
            fixed_json_string = self.fix_json_formatting(json_string)
            try:
                return json.loads(fixed_json_string)
            except json.JSONDecodeError as e:
                print(f"Failed to fix JSON formatting: {e}")
                # Handle the failure case, e.g., return None or raise an error
                return None