import re
import hashlib
import json
import os
import asyncio
from xml.etree.ElementTree import ElementTree
import xml.etree.ElementTree as ET

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, XMLOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.json_handler import JsonHandler

def dict_to_xml(tag, d):
    """
    Convert a dictionary into an XML tree where keys are the XML element tags.
    """
    def build_element(element, dictionary):
        """
        Build an XML element from a dictionary where keys are the XML element tags.
        """
        for key, val in dictionary.items():
            if isinstance(val, ET.Element):
                child = val
            else:
                child = ET.Element(key)
                if isinstance(val, dict):
                    build_element(child, val)
                elif isinstance(val, list):
                    for sub_item in val:
                        if isinstance(sub_item, dict):
                            sub_child = ET.Element(key)
                            build_element(sub_child, sub_item)
                            child.append(sub_child)
                        else:
                            sub_child = ET.Element('item')
                            sub_child.text = str(sub_item)
                            child.append(sub_child)
                else:
                    child.text = str(val)
            element.append(child)

    elem = ET.Element(tag)
    build_element(elem, d)
    return elem

def nest_dict_to_xml(data):
    """
    Helper function dict_to_xml
    Convert a list of dictionaries to an XML tree.
    Handles nested dictionaries and lists.
    """
    final_strings = []
    final_roots = []

    def simple_dict_to_xml(tag, d):
        """
        Turn a simple dict of key/value pairs into XML
        """
        elem = ET.Element(tag)
        for key, val in d.items():
            if isinstance(val, dict):
                child = simple_dict_to_xml(key, val)
                elem.append(child)
            elif isinstance(val, list):
                for sub_item in val:
                    if isinstance(sub_item, dict):
                        sub_elem = simple_dict_to_xml(key, sub_item)
                        elem.append(sub_elem)
            else:
                child = ET.Element(key)
                child.text = str(val).strip()
                elem.append(child)

        return elem

    root = ET.Element('root')
    for item in data:
        if isinstance(item, dict):
            for key, val in item.items():
                if isinstance(val, dict):
                    elem = simple_dict_to_xml(key, val)
                elif isinstance(val, list):
                    elem = ET.Element(key)
                    for sub_item in val:
                        if isinstance(sub_item, dict):
                            sub_elem = simple_dict_to_xml('item', sub_item)
                            elem.append(sub_elem)
                else:
                    elem = ET.Element(key)
                    elem.text = str(val).strip()
                root.append(elem)
            final_strings.append(ET.tostring(root, encoding='unicode'))
            final_roots.append(root)
    
    return final_roots

# Expansion generation
async def generate_expansions(llm, sections, chapter_name, course_name, expansion_length, regions = ["Overview", "Explanations"]):
    def generate_xml_elements(section, elements):
        """
        Generate XML elements for the given list of elements.
        Take this as an example in the prompt.
        """
        section_string = re.sub(r'[^\w\s]', '', str(section))  # Remove any character that is not a word character or whitespace
        section_string = section_string.replace(':', '')  # Explicitly remove colons
        section_string = section_string.replace(' ', '_')  # Replace spaces with underscores
        root = ET.Element(str(section_string))  # Create a root element with the cleaned chapter name
        for element in elements:
            content = f'content for {element} here'
            child = ET.SubElement(root, element)
            child.text = content
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return xml_str
    # output_instructions = generate_xml_elements(regions)

    inputs = [{
                "course_name": course_name,
                "chapter_name": chapter_name,
                "section": section,
                "expansion_length": expansion_length,
                "regions": regions,
                "output_instructions": generate_xml_elements(section, regions),
            } for section in sections]
    # parser = StrOutputParser()
    parser = XMLOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        """
        Course name: {course_name}
        Chapter name: {chapter_name}
        Your task is to provide expansions covering regions: {regions} for the given section: {section}
        Format the output in XML format as follows:
        ----------------
        {output_instructions}
        ----------------
        Max words for expansion: {expansion_length}
        """
    )
    chain = prompt | llm | error_parser
    results = await chain.abatch(inputs)

    final_roots = nest_dict_to_xml(results)

    return dict(zip(sections, final_roots))

# Expansion generation with given number of attempts
def robust_generate_expansions(llm, sections, chapter_name, course_name, expansion_length, max_attempts = 5, regions = ["Overview", "Explanations"]):
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
    results = await chain.abatch(inputs)

    return dict(zip(chapter_list, results))

def robust_generate_sections(llm, zero_shot_topic, chapter_list, sections_per_chapter, max_attempts = 5):
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
        file_path = self.course_meta_dir +  "course_name_textbook_chapters.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.course_name_textbook_chapters = json.load(file)

        else:
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                As as a professor teaching course: {zero_shot_topic}.
                Please work through the following steps:
                1. Find 3 most popular textbooks about this course topic, note down it as ```textbook and author```.
                2. Based on these textbooks, come up with at most 10 and at least 5 learning sessions that the students can learn the entire course step by step.
                3. In chapter name, mark the chapter numbers.
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
        if(self.if_advanced_model):
            llm = self.llm_advance
        else:
            llm = self.llm_basic
        max_note_expansion_words = self.max_note_expansion_words

        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.sections_list = data_temp["sections_list"]
        else:
            self.create_sections()

        for i in range(len(self.chapters_list)):
            if not os.path.exists(self.note_dir + f'notes_set{i}.xml'):
                chapters_name_temp = self.chapters_list[i]
                sections_list_temp = self.sections_list[i]

                if not os.path.exists(self.note_dir + f'notes_set{i}.xml'):
                    notes_exp = robust_generate_expansions(llm, sections_list_temp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], max_note_expansion_words, 5, self.regions)

                    # Convert notes_exp to XML format
                    notes_exp_xml = dict_to_xml('notes_expansion', notes_exp)
                    
                    # Write XML to files
                    tree = ET.ElementTree(notes_exp_xml)
                    ET.indent(tree)

                with open(self.note_dir + f'notes_set_exp{i}.xml', "wb") as f:
                    tree.write(f, encoding="UTF-8", xml_declaration=True)
                with open(self.note_dir + f'notes_set{i}.xml', 'wb') as f:
                    tree.write(f, encoding="UTF-8", xml_declaration=True)

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