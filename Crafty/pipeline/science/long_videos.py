import openai
import json
import os
import numpy as np
import re
import copy
import subprocess
import logging
import fitz
import requests
import multiprocessing
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips

from openai import OpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.json_handler import JsonHandler

# Functions for parsing LaTeX content wihtout titles
def parse_latex_slides(latex_content):
    frame_pattern = re.compile(r'\\begin{frame}.*?\\end{frame}', re.DOTALL)
    # Use the pattern to find all occurrences of frame content in the LaTeX document.
    frames = frame_pattern.findall(latex_content)
    # Initialize an empty list to hold the cleaned text of each frame.
    slide_texts = []
    for frame in frames:
        # Remove LaTeX commands within the frame content.
        # This regex matches LaTeX commands, which start with a backslash followed by any number of alphanumeric characters
        # and may include optional arguments in square or curly braces.
        clean_text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*', '', frame)
        # Further clean the extracted text by removing any leftover curly braces and normalizing whitespace.
        # This includes converting multiple spaces, newlines, and tabs into a single space, and trimming leading/trailing spaces.
        clean_text = re.sub(r'[{}]', '', clean_text)  # Remove curly braces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
        # Append the cleaned text of the current frame to the list of slide texts.
        slide_texts.append(clean_text)
    # Return the list containing the cleaned text of all slides.
    return slide_texts

# Functions for parsing LaTeX content while keeping titles
def parse_latex_slides_raw(latex_content):
    # Compile a regular expression pattern to identify the content of each slide.
    frame_pattern = re.compile(r'\\begin{frame}(.*?)\\end{frame}', re.DOTALL)
    # Use the pattern to find all occurrences of frame content in the LaTeX document.
    frames = frame_pattern.findall(latex_content)
    # Initialize an empty list to hold the modified text of each frame.
    modified_frames = []
    for frame in frames:
        # Convert all symbols to underscores. A symbol is defined as anything that's not a letter, number, or whitespace.
        modified_frame = re.sub(r'[^\w\s]', '_', frame)
        # Append the modified frame text to the list.
        modified_frames.append(modified_frame)
    # Return the list containing the modified text of all slides.
    return modified_frames

# Functions for parsing LaTeX content
def remove_backticks_and_json_strip_replace(input_string):
    # First, strip the backticks from both ends
    stripped_string = input_string.strip("`")
    # Then, replace the "json" part with an empty string
    cleaned_string = stripped_string.replace("json", "", 1)  # The third argument limits the replacement to the first occurrence
    return cleaned_string

# Predefined LaTeX template for creating slides
# Pre-defined function to generate LaTeX template, based on the style selected
def generate_latex_template(style):
    base_template = r"""
    \documentclass{beamer}
    \usepackage[utf8]{inputenc}
    \usepackage{graphicx}
    \usepackage{amsmath, amsfonts, amssymb}

    """
    if style == 'simple':
        template = base_template + r"""
        \usetheme{default}
        \begin{document}
        \title{Lecture Title}
        \author{Author Name}
        \date{\today}
        
        \begin{frame}
        \titlepage
        \end{frame}
        
        \begin{frame}{Slide Title}
        Content goes here.
        \end{frame}
        
        \end{document}
        """
    elif style == 'medium':
        template = base_template + r"""
        \usetheme{Madrid}
        \usecolortheme{whale}
        \begin{document}
        \title{Lecture Title}
        \author{Author Name}
        \date{\today}
        
        \begin{frame}
        \titlepage
        \end{frame}
        
        \begin{frame}{Slide Title}
        Content goes here.
        \end{frame}
        
        \end{document}
        """
    elif style == 'complex':
        template = base_template + r"""
        \usetheme{Berlin}
        \useoutertheme{infolines}
        \usecolortheme{orchid}
        \setbeamertemplate{background canvas}[vertical shading][bottom=white,top=blue!10]
        \begin{document}
        \title{Lecture Title}
        \author{Author Name}
        \date{\today}
        
        \begin{frame}
        \titlepage
        \end{frame}
        
        \begin{frame}{Slide Title}
        Content goes here.
        \end{frame}
        
        \end{document}
        """
    else:
        return "Invalid style selected. Please choose 'simple', 'medium', or 'complex'."
    
    return template

# Load the content of a .tex file from the 'templates' folder given the file name.
def load_tex_content(file_name):
    file_name = file_name + ".tex"
    # Define the path to the 'templates' folder
    folder_path = 'templates'
    # Construct the full path to the file
    # full_path = f'./{folder_path}/{file_name}'
    full_path = "pipeline/templates/" + file_name
    print("\nfull_path for pdf template: ", full_path)
    # Open the file and read its content
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        # If the file is not found, return an informative message
        print("\nFileNotFoundError: ", full_path)
        return f'File {file_name} not found in the templates folder.'
    except Exception as e:
        # For other exceptions, return a message with the error
        return f'An error occurred: {str(e)}'

# Save image from URL to a local folder
def save_image_from_url(url, folder_path, file_name):
    """
    Downloads an image from a given URL and saves it to a specified folder with a specific file name.
    
    Parameters:
    - url (str): The URL of the image to download.
    - folder_path (str): The local folder path where the image should be saved.
    - file_name (str): The name of the file under which the image will be saved.
    
    Returns:
    - str: The path to the saved image file.
    """
    try:
        # Get the image content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
        
        # Construct the full path for the image
        full_path = f"{folder_path}/{file_name}"
        
        # Write the image content to a file in the specified folder
        with open(full_path, 'wb') as image_file:
            image_file.write(response.content)
        
        return full_path
    except requests.RequestException as e:
        return f"An error occurred: {e}"

class Long_videos:
    def __init__(self, para):
        # First, initialize the course_id
        self.course_id = para['course_id']
        print("\nself.course_id: ", self.course_id)
        try:
            self.course_id = para['course_id']
        except KeyError:
            print("Error: 'course_id' is required in parameters.")
            raise

        # Next, initialize the other parameters
        self.slide_max_words = para.get('slide_max_words', 50)
        self.script_max_words = para.get('script_max_words', 100)
        self.slides_template_file = para.get('slides_template_file', "0")
        self.slides_style = para.get('slides_style', "simple")
        self.content_slide_pages = para.get('content_slide_pages', 10)

        # Initialize the API handler for LLMs
        self.llm_basic = ApiHandler(para).models['basic']['instance']
        self.llm_advance = ApiHandler(para).models['advance']['instance']
        self.llm_creative = ApiHandler(para).models['creative']['instance']

        # Create the directories for the lectures
        self.results_dir = para['results_dir']
        self.note_dir = self.results_dir + "notes/" + self.course_id + "/"
        self.course_meta_dir = self.results_dir + "course_meta/" + self.course_id + "/"
        self.long_videos_dir = self.results_dir + "long_videos/" + self.course_id + "/"
        print("\nself.long_videos_dir: ", self.long_videos_dir)
        self.long_videos_dir = os.path.abspath(self.long_videos_dir) + "/"
        print("\nself.long_videos_dir: ", self.long_videos_dir)
        os.makedirs(self.long_videos_dir, exist_ok=True)

        # Initialize the course information, and load the chapters and sections if they exist
        if(os.path.exists(self.course_meta_dir + "course_name_domain.txt")):
            course_meta_file_path = os.path.join(self.course_meta_dir, "course_name_domain.txt")
            if os.path.exists(course_meta_file_path):
                with open(course_meta_file_path, 'r') as file:
                    self.course_description = file.read()
        elif(os.path.exists(self.course_meta_dir + "zero_shot_topic.txt")):
            if(os.path.exists(self.course_meta_dir + "zero_shot_topic.txt")):
                with open(self.course_meta_dir + "zero_shot_topic.txt", 'r') as file:
                    self.course_description = file.read()
        else:
            print("Error: 'course_description' is required in parameters. No file found. Please generate the notes first.")
        print("\nself.course_description: ", self.course_description)

        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.sections_list = data_temp["sections_list"]

    def _load_slides_template(self):
        """
        Loads a LaTeX slides template from a file if specified, or generates a new one based on a given style.
        Returns the LaTeX template content.
        """
        try:
            if self.slides_template_file is None:
                self.slides_template = generate_latex_template(self.slides_style)
            else:
                self.slides_template = load_tex_content(self.slides_template_file)
        except Exception as e:
            logging.error(f"Error loading slides template: {e}")
            self.slides_template = None
        return self.slides_template

    def create_full_slides(self, notes_set_number=-1):
        llm = self.llm_basic

        # Load the chapters and sections if they exist
        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters = data_temp["chapters_list"]
                self.sections = data_temp["sections_list"]
        else:
            print("Error: 'chapters_and_sections.json' not found. Please generate the notes first.")
            return

        self.slides_template = self._load_slides_template()
        directory = self.note_dir + f'notes_set{notes_set_number}.json'
        if os.path.exists(directory):
            with open(directory, 'r') as json_file:
                notes_set = json.load(json_file)
        else:
            notes_set = self.full_notes_set

        # Change the file extension from .txt to .tex
        if os.path.exists(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex"):
            with open(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex", 'r') as file:
                self.full_slides = file.read()
        else:
            # Send the prompt to the API and get a response
            print("\n")
            parser = StrOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            prompt_1 = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                As a professor teaching course: ```{zero_shot_topic}```.
                Based on the provided material: ```{notes_set}```.
                Please follow the following steps and requirements to generate no more than {page_number} pages of slides for chapter ```{chapter}``` of this course.
                Based on the template in latex format ```{tex_template}``` (But keep in mind that this is only a template, so do not need to include the information in it in your response unless it also shows in the provided material.):
                Step 1: Use "Chapter {notes_set_number}: {chapter}" as first page. Specify the chapter number.
                Step 2: Based on the provided material of notes set and chapter topic of this lecture, come out an outline for this lecture and put it as second page.
                        Number of topics should be no more than 5. Topics will correspond to "section" in latex format. Topic names should be short and concise.
                Step 3: Going though the topics of the chapter, generate the slides accordingly.
                        For each topic, generate slides as follows:
                            -> Page 1 to the end of this section:
                            -> Devide this topic into several key concepts.
                            -> Illustrate each one in a separate page frame (instead of subsection).
                            Try to divide the whole illustration into several bullet points and sub-bullet points.
                            -> Then do the same for the next topic (section).
                Step 4: Generate the last 2 pages: one is the summary of this lecture, another one is the "Thank you" page in the end.
                Requirement 1. Do not include any information not included in the provided material of notes set.
                Requirement 2. Focus on illustration of the concepts and do not use figures or tables etc.
                Requirement 3. Try to cover as much information in the provided material as you can.
                """)
            # chain_1 = prompt_1 | llm | error_parser
            chain_1 = LLMChain(llm=self.llm_advance, prompt=prompt_1, output_key="full_slides_temp_1")

            print("\n")
            prompt_2 = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                ```{full_slides_temp_1}``` is the slides in latex format generated for course: ```{zero_shot_topic}```.
                As a professor teaching this course, based on the provided material: ```{notes_set}``` and chapter name: ```{chapter}```.
                Please combine and refine the generated tex file above from step 1 to 4. Make sure your final output follows the following requirements:
                Requirement 0: Do not delete or add any pages from the generated slides.
                Requirement 1. Only response in latex format. This file should be able to be directly compiled, so do not include anything like "```" in response.
                Requirement 2. Do not include any information not included in the provided material of notes set.
                Requirement 3. Focus on illustration of the concepts and do not use figures or tables etc.
                Requirement 4. Try to cover as much information in the provided material as you can.
                """)
            chain_2 = LLMChain(llm=self.llm_advance, prompt=prompt_2, output_key="full_slides_temp_2")

            print("\n")
            prompt_3 = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                ```{full_slides_temp_2}``` are the slides in latex format generated for course: ```{zero_shot_topic}```.
                As a professor teaching this course, based on the provided material: ```{notes_set}``` and chapter name: ```{chapter}```.
                Please refine the generated tex file. Make sure your final output follows the following requirements:
                Requirement 0: Do not delete or add any pages from the generated slides.
                Requirement 1. Only response in latex format. This file should be able to be directly compiled, so do not include anything like "```" in response.
                Requirement 2. Going through each page of the generated slides, make sure each concept is well explained. Add more examples if needed.
                Requirement 3. Make sure the slides as a whole is self-consistent, that means the reader can get all the information from the slides without any missing parts.
                Requirement 4. Recheck the tex format to make sure it is correct as a whole.
                Requirement 5. Build hyperlinks between the outline slide and the corresponding topic slides.
                """)
            chain_3 = LLMChain(llm=self.llm_advance, prompt=prompt_3, output_key="full_slides_temp_3")

            print("\n")
            prompt_4 = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                For latex ```{full_slides_temp_3}``` please check latex grammar and spelling errors. Fix them if any.

                Then for each topic (latex section) in the slides, do the following:
                    -> Page 1: Insert a single blank page with the topic name on top only.
                        instead of ```\begin{{frame}}{{}}
                                        \centering
                                        <topic name>
                                    \end{{frame}}```
                        use ```\begin{{frame}}{{<topic name>}}
                            \end{{frame}}``` as the blank page.
                    -> Page 2 to the end: original pages.
                And do not include anything like "```" in response.
                Reply with the final slides in latex format purely.
                """)
            chain_4 = LLMChain(llm=self.llm_advance, prompt=prompt_4, output_key="full_slides")

            fchain = SequentialChain(chains=[chain_1, chain_2, chain_3, chain_4],\
                                    input_variables=["zero_shot_topic", "notes_set", "page_number", "tex_template", "chapter", "notes_set_number"],\
                                    output_variables=["full_slides_temp_1", "full_slides_temp_2", "full_slides_temp_3", "full_slides"],\
                                    verbose=False)
            response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                    'notes_set': notes_set,\
                                    'page_number': self.content_slide_pages + 2,\
                                    'tex_template': self.slides_template,\
                                    'chapter': self.chapters_list[notes_set_number],\
                                    'notes_set_number': notes_set_number})
            self.full_slides = response["full_slides"]

            if not os.path.exists(self.long_videos_dir +  f"video_description_chapter_{notes_set_number}.json"):
                prompt = ChatPromptTemplate.from_template(
                    """
                    For course ```{zero_shot_topic}``` and chapter ```{chapter}```.
                    Generate a description text for the slides for this lecture within 100 words.
                    Start with "This lecture ..." and make sure the generated content is closely tied to the content of the slide.
                    Lecture slides:
                    ```{full_slides}```
                    """)
                chain = LLMChain(llm=self.llm_basic, prompt=prompt, output_key="video_description")
                fchain = SequentialChain(chains=[chain],\
                                        input_variables=["zero_shot_topic", "chapter", "full_slides"],\
                                        output_variables=["video_description"],\
                                        verbose=False)
                response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                        'chapter': self.chapters_list[notes_set_number],\
                                        'full_slides': self.full_slides})
                with open(self.long_videos_dir +  f"video_description_chapter_{notes_set_number}.json", 'w') as file:
                    json.dump(response["video_description"], file, indent=2)

            # Save the response in a .tex file instead of a .txt file
            with open(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex", 'w') as file:
                file.write(self.full_slides)

    ### How to generate image and use it to refine previous file
    def dalle_image(self, prompt="Crafty", model="dall-e-3", size="1024x1024", quality="standard", notes_set_number=-1, index=0, retry_on_invalid_request=True):
        if(os.path.exists(self.long_videos_dir + f"chapter_{notes_set_number}_dalle_image_{index}.png")):
            return
        prompt_1 = ChatPromptTemplate.from_template(
            """
            For concept: ```{input}``` in course: {zero_shot_topic}, chapter: {chapter}.
            Write a new visual prompt for DALL-E while avoiding any mention of books, signs, titles, text, and words etc.
            Do not include any technical terms, just a simple description.
            Give a graphic description representation of the concept.
            """)
        chain_1 = LLMChain(llm=self.llm_advance, prompt=prompt_1, output_key="prompt")

        fchain = SequentialChain(chains=[chain_1],\
                                input_variables=["input", "zero_shot_topic", "chapter"],\
                                output_variables=["prompt"],\
                                verbose=False)
        response = fchain.invoke({'input': prompt, 'zero_shot_topic': self.course_description, 'chapter': self.chapters_list[notes_set_number]})
        prompt = response["prompt"]

        client = openai.OpenAI()
        try:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )
        except openai.BadRequestError as e:
            if retry_on_invalid_request:
                print(f"OpenAI API request was invalid, retrying with default prompt: {e}")
                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    For course: {zero_shot_topic}, chapter: {chapter}.
                    Write a new visual prompt for DALL-E while avoiding any mention of books, signs, titles, text, and words etc.
                    Do not include any technical terms, just a simple description.
                    Give a graphic description representation of the concept.
                    Since OpenAI API request was invalid for the previous prompt, try to keep the description safe and harmonious.
                    """)
                chain_2 = LLMChain(llm=self.llm_advance, prompt=prompt_2, output_key="prompt")

                fchain = SequentialChain(chains=[chain_2],\
                                        input_variables=["input", "zero_shot_topic", "chapter"],\
                                        output_variables=["prompt"],\
                                        verbose=False)
                response = fchain.invoke({'input': prompt, 'zero_shot_topic': self.course_description, 'chapter': self.chapters_list[notes_set_number]})
                prompt = response["prompt"]
                self.dalle_image(prompt="Crafty", model=model, size=size, quality=quality, notes_set_number=notes_set_number, index=index, retry_on_invalid_request=False)
            else:
                print(f"Retried with default prompt but encountered an error: {e}")
            return
        except openai.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            return
        # If no exceptions, save the image.
        image_url = response.data[0].url
        save_image_from_url(image_url, self.long_videos_dir, f"chapter_{notes_set_number}_dalle_image_{index}.png")

    def insert_images_into_latex(self, notes_set_number):
        latex_file_path = os.path.join(self.long_videos_dir, f"full_slides_for_notes_set{notes_set_number}.tex")
        image_file_pattern = rf"chapter_{notes_set_number}_dalle_image_\d+\.png"
        
        images = [img for img in os.listdir(self.long_videos_dir) if re.match(image_file_pattern, img)]
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        with open(latex_file_path, 'r') as file:
            latex_content = file.readlines()
        
        modified_content, frame_content = [], []
        image_counter = 0
        inside_frame = False
        for line in latex_content:
            if line.strip().startswith("\\begin{frame}"):
                inside_frame, frame_content = True, [line]
            elif line.strip().startswith("\\end{frame}") and inside_frame:
                frame_content.append(line)
                print(f"Processing image {image_counter + 1}")
                print(len(frame_content))
                # Determine if the frame is "empty" by checking its length or other criteria
                if len(frame_content) <= 2 and image_counter < len(images):  # Adjust criteria as needed
                    # Insert image code before the end frame tag
                    frame_content.insert(-1, \
                    f"""\\begin{{figure}}[ht]
                        \\centering
                        \\includegraphics[width=0.55\\textwidth]{{{os.path.join(self.long_videos_dir, images[image_counter])}}}
                    \\end{{figure}}\n""")
                    print(f"Inserted image {images[image_counter]} into frame {image_counter + 1}")
                    image_counter += 1
                modified_content.extend(frame_content)
                inside_frame, frame_content = False, []
            elif inside_frame:
                frame_content.append(line)
            else:
                modified_content.append(line)
        with open(latex_file_path, 'w') as file:
            file.writelines(modified_content)

        if os.path.exists(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex"):
            with open(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex", 'r') as file:
                self.full_slides = file.read()
        if os.path.exists(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".pdf"):
            pass
        else:
            # Define the full path to the .tex file
            tex_file_path = os.path.join(self.long_videos_dir, "full_slides_for_"+f'notes_set{notes_set_number}'+".tex")
            # Set the working directory to the directory containing the .tex file
            working_directory = os.path.dirname(tex_file_path)
            # Your command to run xelatex
            command = ['/Library/TeX/texbin/xelatex', tex_file_path]
            # Run subprocess with cwd set to the directory of the .tex file
            subprocess.run(command, cwd=working_directory)
            subprocess.run(command, cwd=working_directory)

    def tex_image_generation(self, notes_set_number=-1):
        full_slides_images = copy.deepcopy(self.full_slides)
        self.slide_texts_temp = parse_latex_slides(self.full_slides)
        self.slide_texts = parse_latex_slides_raw(self.full_slides)
        for i in range(len(self.slide_texts)):
            if(i >= 2 and self.slide_texts_temp[i] == ''):
                self.dalle_image(prompt=self.slide_texts[i],\
                                notes_set_number=notes_set_number,\
                                index=i)
        print("\n\nself.slide_texts: ", self.slide_texts)

    def create_scripts(self, notes_set_number=-1):
        llm = self.llm_advance
        llm = self.llm_basic
        # Load the chapters and sections if they exist
        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters = data_temp["chapters_list"]
                self.sections = data_temp["sections_list"]
        else:
            print("Error: 'chapters_and_sections.json' not found. Please generate the notes first.")
            return

        self.slides_template = self._load_slides_template()
        directory = self.note_dir + f'notes_set{notes_set_number}.json'
        if os.path.exists(directory):
            with open(directory, 'r') as json_file:
                notes_set = json.load(json_file)
        else:
            notes_set = self.full_notes_set

        # Load in the simple full slides if they exist
        if os.path.exists(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex"):
            with open(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex", 'r') as file:
                self.full_slides = file.read()
        else:
            self.create_full_slides(notes_set_number)
            with open(self.long_videos_dir + "full_slides_for_"+f'notes_set{notes_set_number}'+".tex", 'r') as file:
                self.full_slides = file.read()

        self.slide_texts_temp = parse_latex_slides(self.full_slides)
        print("\nThe content of the slides are: ", self.slide_texts_temp)
        self.slide_texts = parse_latex_slides_raw(self.full_slides)
        print("\n\nNumber of slides pages are:\n\n", len(self.slide_texts))

        # Change the file extension from .tex to .json
        if os.path.exists(self.long_videos_dir + "scripts_for_"+f'notes_set{notes_set_number}'+".json"):
            with open(self.long_videos_dir + "scripts_for_"+f'notes_set{notes_set_number}'+".json", 'r') as json_file:
                self.scripts = json.load(json_file)
        else:
            self.scripts = []
            self.slides = []
            for i in range(len(self.slide_texts)):
                # Send the prompt to the API and get a response
                print("\n")
                # 3. If needed you can refer to the previous context of slides: ```{previous_context}``` as a reference.
                # but this is only for getting smoother transition between slides.
                if(i == 0):
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a brief script for a presentation start with slide: ```{slide_text}``` and ouline: ```{outline}```.
                        No more than 20 words.
                        ----------------------------------------
                        Requirtments:
                        0. Try to be breif and concise.
                        """)
                    chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")

                    print("\n")
                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                        No more than 20 words.
                        ----------------------------------------
                        Requirtments:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        """)
                    chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")

                    fchain = SequentialChain(chains=[chain_1, chain_2],\
                                            input_variables=["zero_shot_topic", "notes_set", "slide_text", "outline", "previous_context", "chapter", "current_script", "script_length"],\
                                            output_variables=["scripts"],\
                                            verbose=False)
                    response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                            'notes_set': notes_set,\
                                            'slide_text': (self.slide_texts)[i],\
                                            'outline': (self.slide_texts)[i+1],\
                                            'previous_context': self.slides,\
                                            'chapter': self.chapters_list[notes_set_number],\
                                            'current_script': self.scripts,\
                                            'script_length': self.script_max_words})
                elif(i == 1):
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a brief script for the outline page in a presentation: ```{slide_text}```.
                        No more than 50 words.
                        ----------------------------------------
                        Requirtments:
                        0. Try to be breif and concise.
                        """)
                    chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")

                    print("\n")
                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                        No more than 50 words.
                        ----------------------------------------
                        Requirtments:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        """)
                    chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")

                    fchain = SequentialChain(chains=[chain_1, chain_2],\
                                            input_variables=["zero_shot_topic", "notes_set", "slide_text", "outline", "previous_context", "chapter", "current_script", "script_length"],\
                                            output_variables=["scripts"],\
                                            verbose=False)
                    response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                            'notes_set': notes_set,\
                                            'slide_text': (self.slide_texts)[i],\
                                            'outline': (self.slide_texts)[i+1],\
                                            'previous_context': self.slides,\
                                            'chapter': self.chapters_list[notes_set_number],\
                                            'current_script': self.scripts,\
                                            'script_length': self.script_max_words})
                elif(i == len(self.slide_texts) - 1):
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a brief script (1 or 2 sentences) for a presentation end with slide: ```{slide_text}```.
                        Try to be open and inspiring students to think and ask questions.
                        """)
                    chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")

                    print("\n")
                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}``` in only 1 or 2 sentences..
                        ----------------------------------------
                        Requirtments:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        """)
                    chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")

                    fchain = SequentialChain(chains=[chain_1, chain_2],\
                                            input_variables=["zero_shot_topic", "notes_set", "slide_text", "previous_context", "chapter", "current_script", "script_length"],\
                                            output_variables=["scripts"],\
                                            verbose=False)
                    response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                            'notes_set': notes_set,\
                                            'slide_text': (self.slide_texts)[i],\
                                            'previous_context': self.slides,\
                                            'chapter': self.chapters_list[notes_set_number],\
                                            'current_script': self.scripts,\
                                            'script_length': self.script_max_words})
                elif(i == len(self.slide_texts) - 2):
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a brief script for the summarizing slide: ```{slide_text}```.
                        As a reference, the outline of this lecture is: ```{outline}```.
                        No more than 50 words.
                        ----------------------------------------
                        Requirtments:
                        0. Try to be open and inspiring students to think and ask questions.
                        """)
                    chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")

                    print("\n")
                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                        No more than 50 words.
                        ----------------------------------------
                        Requirtments:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        """)
                    chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")

                    fchain = SequentialChain(chains=[chain_1, chain_2],\
                                            input_variables=["zero_shot_topic", "notes_set", "slide_text", "outline", "previous_context", "chapter", "current_script", "script_length"],\
                                            output_variables=["scripts"],\
                                            verbose=False)
                    response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                            'notes_set': notes_set,\
                                            'slide_text': (self.slide_texts)[i],\
                                            'outline': (self.slide_texts)[i+1],\
                                            'previous_context': self.slides,\
                                            'chapter': self.chapters_list[notes_set_number],\
                                            'current_script': self.scripts,\
                                            'script_length': self.script_max_words})
                elif(i != 0 and i != 1 and i != len(self.slide_texts) - 1 and i != len(self.slide_texts) - 2):
                    if(len(self.slide_texts_temp[i]) < 5):
                        prompt_1 = ChatPromptTemplate.from_template(
                            """
                            As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                            Please generate a script for the slide: ```{slide_text}```.
                            Since the slide is a slide with only a title, please generate a brief script around the title to give an overview with 1 or 2 sentences.
                            As a reference, the content of next slide is: ```{next_slide_text}```.
                            ----------------------------------------
                            Requirtments:
                            0. All the information in the slide has been covered.
                            1. The content must be only relevant to the content: ```{slide_text}``` in this specific slide.
                            2. Provide rich examples and explanations and possible applications for the content when needed.
                            3. The response should be directly talking about the academic content, with no introduction or conclusion (like "Today...", or "Now...", "In a word...").
                            """)
                        chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")
                        print("\n")
                        prompt_2 = ChatPromptTemplate.from_template(
                            """
                            As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                            Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                            Since the slide is a slide with only a title, please generate a brief script around the title to give an overview with 1 or 2 sentences.
                            ----------------------------------------
                            Requirtments:
                            0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                            """)
                        chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")
                        fchain = SequentialChain(chains=[chain_1, chain_2],\
                                                input_variables=["zero_shot_topic", "notes_set", "slide_text", "next_slide_text", "previous_context", "chapter", "current_script", "script_length"],\
                                                output_variables=["scripts"],\
                                                verbose=False)
                        response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                                'notes_set': notes_set,\
                                                'slide_text': (self.slide_texts)[i],\
                                                'next_slide_text': (self.slide_texts)[i+1],\
                                                'previous_context': self.slides,\
                                                'chapter': self.chapters_list[notes_set_number],\
                                                'current_script': self.scripts,\
                                                'script_length': self.script_max_words})
                    else:
                        prompt_1 = ChatPromptTemplate.from_template(
                            """
                            As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                            Please generate a script for the slide: ```{slide_text}```.
                            Do not talk about the title of this slide. Just focus on the content.
                            Keep in mind that in the previous slide, the basic idea of the concept illustrated in this slide has been introduced.
                            So do not even talk about the defination of this concept. Just focus on the content and explain the content in the slide.
                            ----------------------------------------
                            Requirtments:
                            0. All the information in the slide has been covered.
                            1. The content must be only relevant to the content: ```{slide_text}``` in this specific slide.
                            2. Provide rich examples and explanations and possible applications for the content when needed.
                            3. The response should be directly talking about the academic content, with no introduction or conclusion (like "Today...", or "Now...", "In a word...").
                            """)
                        chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="scripts_temp_1")
                        print("\n")
                        prompt_2 = ChatPromptTemplate.from_template(
                            """
                            As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                            Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                            Keep in mind that in the previous slide, the basic idea of the concept illustrated in this slide has been introduced.
                            So do not even talk about the defination of this concept. Just focus on the content and explain the content in the slide.
                            ----------------------------------------
                            Requirtments:
                            0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                            1. Remove the first sentence if it is not directly talking about the academic content.
                            """)
                        chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="scripts")
                        fchain = SequentialChain(chains=[chain_1, chain_2],\
                                                input_variables=["zero_shot_topic", "notes_set", "slide_text", "previous_context", "chapter", "current_script", "script_length"],\
                                                output_variables=["scripts"],\
                                                verbose=False)
                        response = fchain.invoke({'zero_shot_topic': self.course_description,\
                                                'notes_set': notes_set,\
                                                'slide_text': (self.slide_texts)[i],\
                                                'previous_context': self.slides,\
                                                'chapter': self.chapters_list[notes_set_number],\
                                                'current_script': self.scripts,\
                                                'script_length': self.script_max_words})

                self.scripts.append(response["scripts"])
                self.slides.append((self.slide_texts)[i])
                # self.scripts.append(ast.literal_eval(remove_backticks_and_json_strip_replace(response["scripts"])))

            with open(self.long_videos_dir + "scripts_for_"+f'notes_set{notes_set_number}'+".json", 'w') as file:
                json.dump(self.scripts, file, indent=2)

    def voice_agent(self, speech_file_path=None, input_text=None, model="tts-1", voice="alloy", notes_set_number=-1):
        """
        Generates an audio speech file from the given text using the specified voice and model, with a 1-second silent time after the content.
        
        :param speech_file_path: The path where the audio file will be saved. If None, saves to a default directory.
        :param input_text: The text to be converted to speech. If None, a default phrase will be used.
        :param model: The text-to-speech model to use.
        :param voice: The voice model to use for the speech.
        """
        if input_text is None:
            input_text = "input_text not defined"
        
        if speech_file_path is None:
            speech_file_path = self.long_videos_dir + f"voice_{-1}_chapter_{notes_set_number}.mp3"

        try:
            # Generate the speech audio
            response = OpenAI().audio.speech.create(
                model=model,
                voice=voice,
                input=input_text
            )
            
            # Save the generated speech to a temporary file
            temp_audio_file = f"temp_speech_{notes_set_number}.mp3"
            with open(temp_audio_file, "wb") as f:
                f.write(response.content)
            
            # Load the speech audio and create a 1-second silence
            speech_audio = AudioSegment.from_file(temp_audio_file)
            one_second_silence = AudioSegment.silent(duration=2000)  # 1,000 milliseconds

            # Combine speech audio with silence
            final_audio = speech_audio + one_second_silence

            # Save the combined audio
            final_audio.export(speech_file_path, format="mp3", parameters=["-ar", "16000"])
            
            # Clean up the temporary file
            os.remove(temp_audio_file)
        except Exception as e:
            print(f"Failed to generate audio: {e}")
            return None

        return speech_file_path
    
    def scripts2voice(self, speech_file_path=None, input_text=None, model="tts-1", voice="alloy", notes_set_number=-1):
        """
        Converts scripts into mp3 files. If the script files do not exist, it creates all necessary components.
        """
        scripts_file_path = f"{self.long_videos_dir}scripts_for_notes_set{notes_set_number}.json"
        
        if not os.path.exists(scripts_file_path):
            self.create_scripts(notes_set_number)

        with open(scripts_file_path, 'r') as json_file:
            self.scripts = json.load(json_file)  #["scripts"]

        for i, script in enumerate(self.scripts):
            voice_file_path = speech_file_path if speech_file_path and (speech_file_path.endswith("/") and os.path.exists(speech_file_path)) else self.long_videos_dir
            voice_file_path += f"voice_{i}_chapter_{notes_set_number}.mp3"
            # print("\n\nCurrent script is: ", script)
            if not os.path.exists(voice_file_path):
                self.voice_agent(speech_file_path=voice_file_path, input_text=str(script), model=model, voice=voice)

    ### With voice generated, create small video for each slide
    def pdf2image(self, notes_set_number=-1):
        pdf_file_path = self.long_videos_dir + f"full_slides_for_notes_set{notes_set_number}.pdf"
        doc = fitz.open(pdf_file_path)
        
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            
            # Increase the dpi by adjusting the zoom factor. Default is 1.0 (72 dpi).
            # For higher resolution, you might use 2.0 (144 dpi) or higher.
            zoom = 8.0  # Adjust this factor to get higher resolution images.
            mat = fitz.Matrix(zoom, zoom)  # The transformation matrix for scaling.
            
            pix = page.get_pixmap(matrix=mat)  # Use the matrix in get_pixmap
            image_path = self.long_videos_dir + f"image_{page_number}_chapter_{notes_set_number}.png"
            pix.save(image_path)

    def mp3_to_mp4_and_combine(self, notes_set_number):
        """
        Converts MP3 files into MP4 files using corresponding images as static backgrounds,
        sets a default frame rate (fps) for the video, and combines all MP4 files into one,
        skipping already existing MP4 files and the final combination if it exists, for a specific chapter number.
        
        :param output_dir: Directory where the MP3 files, PNG files, MP4 files, and the final combined MP4 file are located.
        :param notes_set_number: Specific chapter number to match voice and image files.
        """
        output_dir = self.long_videos_dir
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the name of the final combined video file
        final_output_filename = f"combined_video_chapter_{notes_set_number}.mp4"
        final_output_path = os.path.join(output_dir, final_output_filename)
        
        # Check if the combined MP4 file already exists
        if os.path.exists(final_output_path):
            print(f"Combined video {final_output_path} already exists, skipping combination.")
            return  # Exit the function if combined video already exists

        # List all MP3 files and sort them by the index i for the specific chapter
        chapter_str = f"_chapter_{notes_set_number}"
        audio_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.mp3') and chapter_str in f],
                            key=lambda x: int(x.split('_')[1]))

        # List to hold all the individual video clips
        video_clips = []

        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            output_mp4_path = os.path.join(output_dir, f"{base_name}.mp4")
            
            # Check if MP4 file already exists to avoid re-generating it
            if not os.path.exists(output_mp4_path):
                image_file = f"{base_name.replace('voice_', 'image_')}.png"
                audio_path = os.path.join(output_dir, audio_file)
                image_path = os.path.join(output_dir, image_file)
                
                if os.path.exists(image_path) and os.path.exists(audio_path):
                    # Load the audio file
                    audio_clip = AudioFileClip(audio_path)
                    
                    # Create an image clip with the same duration as the audio file
                    image_clip = ImageClip(image_path).set_duration(audio_clip.duration)
                    
                    # Set the audio of the image clip as the audio file
                    video_clip = image_clip.set_audio(audio_clip)
                    
                    # Write the individual video clip to a file (MP4)
                    video_clip.write_videofile(output_mp4_path, codec="libx264", audio_codec="aac", fps=12)
                    print(f"Generated {output_mp4_path}")
                else:
                    print(f"Missing files for {base_name}, cannot generate MP4.")
                    continue  # Skip to the next file if either file is missing
            else:
                print(f"MP4 file {output_mp4_path} already exists, skipping generation.")

            # Load the existing or newly created MP4 file for final combination
            video_clips.append(VideoFileClip(output_mp4_path))

        # Combine all the video clips into one video file
        if video_clips:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=12)
            print(f"Generated combined video {final_output_path}")
        else:
            print("No video clips to combine.")

    def create_long_videos(self, chapter=0):
        self.create_full_slides(notes_set_number = chapter)  #"notes_set1"
        self.create_scripts(notes_set_number = chapter)  #"notes_set1"
        self.tex_image_generation(notes_set_number = chapter)
        self.scripts2voice(notes_set_number = chapter)
        self.insert_images_into_latex(notes_set_number = chapter)
        self.pdf2image(notes_set_number = chapter)
        self.mp3_to_mp4_and_combine(notes_set_number = chapter)

class VideoProcessor:
    def __init__(self, para):
        self.para = para
        self.course_id = para['course_id']
        print("\nself.course_id: ", self.course_id)
        try:
            self.course_id = para['course_id']
        except KeyError:
            print("Error: 'course_id' is required in parameters.")
            raise
        # Create the directories for the lectures
        self.results_dir = para['results_dir']
        self.note_dir = self.results_dir + "notes/" + self.course_id + "/"
        self.course_meta_dir = self.results_dir + "course_meta/" + self.course_id + "/"
        self.long_videos_dir = self.results_dir + "long_videos/" + self.course_id + "/"
        print("\nself.long_videos_dir: ", self.long_videos_dir)
        self.long_videos_dir = os.path.abspath(self.long_videos_dir) + "/"
        print("\nself.long_videos_dir: ", self.long_videos_dir)
        os.makedirs(self.long_videos_dir, exist_ok=True)
        if os.path.exists(self.note_dir + "chapters_and_sections.json"):
            with open(self.note_dir + "chapters_and_sections.json", 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.sections_list = data_temp["sections_list"]

    def worker_process(self, chapter):
        """ Worker function to initialize Long_videos and process each chapter """
        long_video = Long_videos(self.para)
        long_video.create_long_videos(chapter)

    def run_parallel_processing(self, number_of_trials=2):
        """ Process each chapter in parallel using multiprocessing """
        for _ in range(number_of_trials):
            if hasattr(self, 'chapters_list') and self.chapters_list:
                with multiprocessing.Pool() as pool:
                    # Create tasks for each chapter
                    for chapter in range(len(self.chapters_list)):
                        pool.apply_async(self.worker_process, args=(chapter,))
                    pool.close()  # No more tasks will be submitted to the pool
                    pool.join()  # Wait for all the tasks to complete
            else:
                print("Error: 'chapters_list' not properly initialized.")

    def run_sequential_processing(self):
        """ Process each chapter sequentially """
        for chapter in range(len(self.chapters_list)):
            self.worker_process(chapter)