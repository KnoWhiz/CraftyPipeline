import hashlib
import json
import os
import pandas as pd
import time
import asyncio

# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnableParallel

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.json_handler import JsonHandler

# Helper functions
def format_list_to_string(strings):
    """
    Example:
    example_list = ["example 1", "example 2", "example 3", "example 4"]
    formatted_string = format_list_to_string(example_list)
    print(formatted_string)
    """
    return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(strings))

# Helper functions
def format_expansion(data):
    """
    Processes the given data to convert any lists into formatted strings.
    Example:
    with open("1.json", 'r') as json_file:
        data_temp = json.load(json_file)
    formatted_data = format_expansion(data_temp)
    with open("1.json", 'w') as json_file:
        json.dump(formatted_data, json_file, indent=4)
    """
    def convert_lists(item):
        if isinstance(item, list):
            return format_list_to_string(item)
        elif isinstance(item, dict):
            return {key: convert_lists(value) for key, value in item.items()}
        else:
            return item
    return {key: convert_lists(value) for key, value in data.items()}

# Expansion generation
async def generate_expansions(llm, sections, chapter_name, course_name, expansion_length, regions = ["Overview", "Explanations"]):
    def format_string(regions):
        json_content = "\n".join([f'\"{region}\": \"<content of {region} for the given section here>\",' for region in regions])
        json_format_string = f"""
        ----------------
        ```json
        {{
            {json_content.rstrip(',')}
        }}
        ```
        ----------------
        """
        return json_format_string
    json_format_string = format_string(regions)
    inputs = [{
                "course_name": course_name,
                "chapter_name": chapter_name,
                "section": section,
                "expansion_length": expansion_length,
                "json_format_string": json_format_string,
            } for section in sections]
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        """
        Course name: {course_name}
        Chapter name: {chapter_name}
        Your task is to provide an expansion for the given section.
        section: {section}
        Max words for expansion: {expansion_length}
        It should formated as json:
        {json_format_string}
        """
    )
    chain = prompt | llm | error_parser
    # asyncio.run(chain.abatch(inputs))
    results = await chain.abatch(inputs)

    return dict(zip(sections, results))

# Expansion generation with given number of attempts
def robust_generate_expansions(llm, sections, chapter_name, course_name, expansion_length, max_attempts = 3, regions = ["Overview", "Explanations"]):
    attempt = 0
    while attempt < max_attempts:
        try:
            return asyncio.run(generate_expansions(llm, sections, chapter_name, course_name, expansion_length, regions))
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for generating expansions: {e}")
            attempt += 1
            if attempt == max_attempts:
                print(f"Failed to generate expansions after {max_attempts} attempts.")
                # Return None or raise an exception depending on how you want to handle complete failure.
                raise Exception(f"Expansions generation failed after {max_attempts} attempts.")

# sections generation
async def generate_sections(llm, zero_shot_topic, chapter_list, sections_per_chapter):
    inputs = [{
                "zero_shot_topic": zero_shot_topic,
                "chapter_name": chapter,
                "sections_per_chapter": sections_per_chapter,
            } for chapter in chapter_list]
    
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        """
        Requirements: \n\n\n
        As as a professor teaching course: {zero_shot_topic}.
        Please work through the following steps:
        Come up with the sections in the chapter: {chapter_name}.
        Number of sections within the chapter should be no more than: {sections_per_chapter} and no less than 5.
        The output format should be:
        ----------------
        ```json
        {{
        "sections": [
            <section_1>,
            <section_2>,
            ...
            <section_n>,
        ]
        }}
        ```
        ----------------
        """
    )
    chain = prompt | llm | error_parser
    # asyncio.run(chain.abatch(inputs))
    results = await chain.abatch(inputs)

    return dict(zip(chapter_list, results))

def robust_generate_sections(llm, zero_shot_topic, chapter_list, sections_per_chapter, max_attempts = 3):
    attempt = 0
    while attempt < max_attempts:
        try:
            return asyncio.run(generate_sections(llm, zero_shot_topic, chapter_list, sections_per_chapter))
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for generating sections: {e}")
            attempt += 1
            if attempt == max_attempts:
                print(f"Failed to generate sections after {max_attempts} attempts.")
                # Return None or raise an exception depending on how you want to handle complete failure.
                raise Exception(f"sections generation failed after {max_attempts} attempts.")

class Zeroshot_notes:
    def __init__(self, para):
        self.regions = para["regions"]
        self.if_advanced_model = para["if_advanced_model"]
        self.course_info = para["course_info"]
        self.results_dir = para['results_dir']
        self.sections_per_chapter = para['sections_per_chapter']
        self.max_note_expansion_words = para['max_note_expansion_words']
        self.llm_basic = ApiHandler(para).models['basic']['instance']
        self.llm_advance = ApiHandler(para).models['advance']['instance']
        self._hash_course_info()
        self.note_dir = self.results_dir + "notes/" + self.course_id + "/"
        self.quiz_dir = self.results_dir + "quiz/" + self.course_id + "/"
        self.test_dir = self.results_dir + "test/" + self.course_id + "/"
        self.course_meta_dir = self.results_dir + "course_meta/" + self.course_id + "/"

        self.notes_list = []
        os.makedirs(self.note_dir, exist_ok=True)
        os.makedirs(self.quiz_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.course_meta_dir, exist_ok=True)

        print("\nself.course_meta_dir: ", self.course_meta_dir)
        self._extract_zero_shot_topic()

    def _hash_course_info(self):
        """
        Hash the course description.
        """
        # Initialize a hashlib object for SHA-224
        sha224_hash = hashlib.sha224()
        sha224_hash.update(self.course_info.encode("utf-8"))

        # Calculate the final hash
        self.course_id = sha224_hash.hexdigest()

    def _extract_zero_shot_topic(self):
        """
        Get the zero_shot_topic based on self.course_info = para["course_info"].
        """
        llm = self.llm_advance
        llm = self.llm_basic
        if(os.path.exists(self.course_meta_dir + "zero_shot_topic.txt")):
            with open(self.course_meta_dir + "zero_shot_topic.txt", 'r') as file:
                self.zero_shot_topic = file.read()
        else:
            # Support complex input formats
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                Based on the information of the course information about the course that a student wants to learn: ```{course_info}```.
                "Context" is a restrictive description of the course,
                and "subject" is the general topic of the course,
                and "text" is the detailed description about the content that this user wants to learn.
                Please answer: what is the zero_shot_topic of this course should be by combining "context", "subject", and "text".
                For example, input can be like this:
                ```
                context: "Bayesian"
                level: "Beginner"
                subject: "Machine learning"
                text: "Bayesian machine learning techniques"
                ```
                The response should be formated as json:
                ```json
                {{
                "context": <what is the context of this course>,
                "level": <what is the level of this course>,
                "subject": <what is the subject of this course>,
                "zero_shot_topic": <what is the zero_shot_topic of this course>
                }}
                ```
                """
            )
            chain = prompt | llm | error_parser
            response = chain.invoke({'course_info': self.course_info})
            self.zero_shot_topic = response["zero_shot_topic"]
            self.level = response["level"]
            with open(self.course_meta_dir + "zero_shot_topic.txt", 'w') as file:
                file.write(self.zero_shot_topic)

    def create_chapters(self):
        llm = self.llm_advance
        # llm = self.llm_basic
        file_path = self.course_meta_dir +  "course_name_textbook_chapters.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.course_name_textbook_chapters = json.load(file)
                # self.course_name_textbook_chapters = file.read()

        else:
            # Send the prompt to the API and get response
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                As as a professor teaching course: {zero_shot_topic}.
                Please work through the following steps:
                1. Find 3 most popular textbooks about this course topic, note down it as ```textbook and author```.
                2. Based on these textbooks, come up with at most 10 and at least 5 learning sessions that the students can learn the entire course step by step.
                The output format should be json as follows:
                ```json
                {{
                "Course name": <course name here>,

                "Textbooks": [
                    <textbook_1 here>,
                    <textbook_2 here>,
                    <textbook_3 here>,
                ]

                "authors": [
                    <author_1 here>,
                    <author_2 here>,
                    <author_3 here>,
                ]

                "Chapters": [
                    <chapter_1>,
                    <chapter_2>,
                    ...
                    <chapter_n>,
                ]
                }}
                ```
                """)
            chain = prompt | llm | error_parser
            response = chain.invoke({'zero_shot_topic': self.zero_shot_topic})
            self.course_name_textbook_chapters = response
            file_path = self.course_meta_dir +  "course_name_textbook_chapters.json"
            with open(file_path, 'w') as file:
                json.dump(self.course_name_textbook_chapters, file, indent=2)

    def create_sections(self):
        llm = self.llm_advance
        # llm = self.llm_basic
        self.create_chapters()

        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.sections_list = data_temp["sections_list"]
        else:
            with open(self.course_meta_dir + "course_name_textbook_chapters.json", 'r') as file:
                self.chapters_list = json.load(file)["Chapters"]
                self.raw_sections_in_chapters = []
                self.raw_sections_in_chapters = robust_generate_sections(llm, self.zero_shot_topic, self.chapters_list, self.sections_per_chapter)

                self.sections_list = []

                parser = JsonOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    Based on {raw_sections_in_chapters}, the sections in a list of lists. The length of the list should be the same as the number of chapters.
                    Make sure every section is unique: If one section has a similar meaning with another section in another chapter,
                    only keep the first one (with lower chapter index) and remove the other sections.
                    Chapter list: ```{chapters_list}```.
                    Use the following json format:
                    ----------------
                    {{
                    "sections": [
                        [<section_1>, <section_2>, ..., <section_n>],
                        [<section_1>, <section_2>, ..., <section_m>],
                        ...
                        [<section_1>, <section_2>, ..., <section_p>],
                    ]
                    }}
                    ----------------
                    """
                )
                chain = prompt | llm | error_parser
                response = chain.invoke({'chapters_list': self.chapters_list, 'raw_sections_in_chapters': self.raw_sections_in_chapters})
                self.sections_list = response["sections"]

                with open(self.course_meta_dir +  "raw_sections_in_chapters.json", 'w') as file:
                    json.dump(self.raw_sections_in_chapters, file, indent=2)
                with open(self.course_meta_dir +  "chapters_list.json", 'w') as file:
                    json.dump(self.chapters_list, file, indent=2)
                with open(self.course_meta_dir +  "sections_list.json", 'w') as file:
                    json.dump(self.sections_list, file, indent=2)

            data_temp = {
                "chapters_list": self.chapters_list,
                "sections_list": self.sections_list
            }

            # Save to JSON file
            with open(self.note_dir + "chapters_and_sections.json", 'w') as json_file:
                json.dump(data_temp, json_file, indent=4)
        return data_temp

    def create_notes(self):
        # llm = self.llm_advance
        if(self.if_advanced_model):
            llm = self.llm_advance
        else:
            llm = self.llm_basic
        max_note_expansion_words = self.max_note_expansion_words
        full_notes_set = []

        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.sections_list = data_temp["sections_list"]
        else:
            self.create_sections()

        try:
            # TODO: Try to generate the full notes set
            raise Exception("Test exception, will replace with actual code")
        except Exception as e:
            # If the full notes set generation fails, go back to the previous steps and try again
            print(f"Error generating full notes set: {e}") # Log the error
            for i in range(len(self.chapters_list)):
                if os.path.exists(self.note_dir +  f'notes_set{i}.json'):
                    with open(self.note_dir + f'notes_set{i}.json', 'r') as file:
                        notes = json.load(file)
                        # notes_set_temp = json.load(file)
                        full_notes_set.append(notes)
                else:
                    chapters_name_temp = self.chapters_list[i]
                    sections_list_temp = self.sections_list[i]

                    if os.path.exists(self.note_dir + f'notes_set_exp{i}.json'):
                        with open(self.note_dir + f'notes_set_exp{i}.json', 'r') as file:
                            notes_exp = json.load(file)
                    else:
                        try:
                            notes_exp = robust_generate_expansions(llm, sections_list_temp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], max_note_expansion_words, 3, self.regions)
                        except Exception as e:
                            print(f"Error generating expansions for chapter {chapters_name_temp}: {e}")
                            continue  # Skip this iteration and proceed with the next chapter

                    notes = format_expansion(notes_exp)
                    full_notes_set.append(notes)
                    with open(self.note_dir + f'notes_set_exp{i}.json', 'w') as file:
                        json.dump(notes_exp, file, indent=2)
                    with open(self.note_dir + f'notes_set{i}.json', 'w') as file:
                        json.dump(notes, file, indent=2)

        with open(self.note_dir + f'full_notes_set.json', 'w') as file:
            json.dump(full_notes_set, file, indent=2)

        self.full_notes_set = full_notes_set
        return full_notes_set

    def get_chapters_notes_list(self):
        return self.full_notes_set

    def get_all_notes_list(self):
        all_notes = {k: v for d in self.full_notes_set for k, v in d.items()}
        return all_notes

    def get_chapters_list(self):
        return self.chapters_list

    def get_hash_id(self):
        return self.course_id

    def get_course_name(self):
        if "Course name" in self.course_name_textbook_chapters:
            return self.course_name_textbook_chapters["Course name"]
        else:
            return ""