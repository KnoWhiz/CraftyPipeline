import json
import os

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.pipeline.pipeline_step import PipelineStep
from Crafty.pipeline.utils.tex import TexUtil


class Script(PipelineStep):

    def __init__(self, para):
        super().__init__(para)
        self.llm = self.llm_advance

        self.short_video = para['short_video']
        self.zero_shot_topic = para['topic']
        self.chapters_list = [self.zero_shot_topic]

        if(self.short_video == True):
            # self.chapter = 0
            self.chapter = para['chapter']
        else:
            self.chapter = para['chapter']
            self.read_meta_data_from_file()

    def execute(self):
        if self.chapter is None or self.chapter < 0:
            raise ValueError("Chapter number is not provided or invalid.")

        if(self.short_video == True):
            self.create_scripts_short(self.chapter)
        else:
            self.create_scripts(self.chapter)

    def create_scripts_short(self, notes_set_number=-1):
        """
        Generate scripts for short videos.
        """
        directory = self.notes_dir + f'notes_set{notes_set_number}.xml'
        if os.path.exists(directory):
            with open(directory, 'r') as xml_file:
                notes_set = xml_file.read()

        # Load in the simple full slides if they exist
        if os.path.exists(self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex"):
            with open(self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex", 'r') as file:
                full_slides = file.read()
        else:
            raise FileNotFoundError(f"Full slides file not found in {self.videos_dir}")

        slide_texts_temp = TexUtil.parse_latex_slides(full_slides)
        click.echo(f"The content of the slides are: {slide_texts_temp}")
        slide_texts = TexUtil.parse_latex_slides_raw(full_slides)
        click.echo(f"Number of slides pages are: {len(slide_texts)}")

        chapter_scripts = []
        slides = []

        print("len(slide_texts): ", len(slide_texts))

        for i in range(len(slide_texts)):
            if (i == 0):
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching course {zero_shot_topic}.
                    Please generate a brief script for the first slide page: ```{slide_text}```.

                    Requirements:
                    1. Reply a fluent sentence with no more than 20 words.
                    2. Try to be brief and concise.
                    3. The response only has the information related to the slide.
                    4. No more pleasantries
                    """)
                chain = prompt | self.llm | error_parser
                scripts = chain.invoke({'zero_shot_topic': self.zero_shot_topic,
                                        'notes_set': notes_set,
                                        'slide_text': slide_texts[i],
                                        'chapter': self.chapters_list[notes_set_number]})

            else:
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching course {zero_shot_topic}.
                    Please generate a brief script for a content slide page: ```{slide_text}```.
                    
                    Requirements:
                    1. Reply a fluent sentence with no more than 20 words.
                    2. Try to be brief and concise.
                    3. The response only has the information related to the slide.
                    4. No more pleasantries
                    """)
                chain = prompt | self.llm | error_parser
                scripts = chain.invoke({'zero_shot_topic': self.zero_shot_topic,
                                        'notes_set': notes_set,
                                        'slide_text': slide_texts[i],
                                        'chapter': self.chapters_list[notes_set_number]})
                
            chapter_scripts.append(scripts)
            slides.append(slide_texts[i])
            click.echo(f"Scripts generated for slide {i}")

        file_path = self.videos_dir + f'scripts_for_notes_set{notes_set_number}' + ".json"
        with open(file_path, 'w') as file:
            json.dump(chapter_scripts, file, indent=2)
        click.echo(f"Scripts for note set {notes_set_number} are saved to: {file_path}")

    def create_scripts(self, notes_set_number=-1):
        """
        Generate scripts for each slide in the full slides LaTeX file based on the notes set number.
        The scripts are generated by calling the LLM model with a specific prompt for each slide (depending on the position of that slide).
        """

        directory = self.notes_dir + f'notes_set{notes_set_number}.xml'
        if os.path.exists(directory):
            with open(directory, 'r') as xml_file:
                notes_set = xml_file.read()

        # Load in the simple full slides if they exist
        if os.path.exists(self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex"):
            with open(self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex", 'r') as file:
                full_slides = file.read()
        else:
            raise FileNotFoundError(f"Full slides file not found in {self.videos_dir}")

        slide_texts_temp = TexUtil.parse_latex_slides(full_slides)
        click.echo(f"The content of the slides are: {slide_texts_temp}")
        slide_texts = TexUtil.parse_latex_slides_raw(full_slides)
        click.echo(f"Number of slides pages are: {len(slide_texts)}")

        chapter_scripts = []
        slides = []

        for i in range(len(slide_texts)):
            # Send the prompt to the API and get a response
            # 3. If needed you can refer to the previous context of slides: ```{previous_context}``` as a reference.
            # but this is only for getting smoother transition between slides.
            if (i == 0):
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt_1 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please generate a brief script for a presentation start with slide: ```{slide_text}``` and ouline: ```{outline}```.
                    No more than 20 words.
                    ----------------------------------------
                    Requirements:
                    0. Try to be brief and concise.
                    """)
                chain_1 = prompt_1 | self.llm | error_parser
                scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                 'notes_set': notes_set,
                                                 'slide_text': slide_texts[i],
                                                 'outline': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                                 'chapter': self.chapters_list[notes_set_number]})


                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                    No more than 20 words.
                    ----------------------------------------
                    Requirements:
                    0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                    """)
                chain_2 = prompt_2 | self.llm | error_parser
                scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'notes_set': notes_set,
                                          'slide_text': slide_texts[i],
                                          'outline': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                          'chapter': self.chapters_list[notes_set_number],
                                          'scripts_temp_1': scripts_temp_1})

            elif (i == 1):
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt_1 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please generate a brief script for the outline page in a presentation: ```{slide_text}```.
                    No more than 50 words.
                    ----------------------------------------
                    Requirements:
                    0. Try to be brief and concise.
                    """)
                chain_1 = prompt_1 | self.llm | error_parser
                scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                 'notes_set': notes_set,
                                                 'slide_text': slide_texts[i],
                                                 'chapter': self.chapters_list[notes_set_number]})


                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                    No more than 50 words.
                    ----------------------------------------
                    Requirements:
                    0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                    """)
                chain_2 = prompt_2 | self.llm | error_parser
                scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'notes_set': notes_set,
                                          'slide_text': slide_texts[i],
                                          'chapter': self.chapters_list[notes_set_number],
                                          'scripts_temp_1': scripts_temp_1})

            elif i == len(slide_texts) - 1:
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt_1 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please generate a brief script (1 or 2 sentences) for a presentation end with slide: ```{slide_text}```.
                    Try to be open and inspiring students to think and ask questions.
                    """)
                chain_1 = prompt_1 | self.llm | error_parser
                scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                 'notes_set': notes_set,
                                                 'slide_text': slide_texts[i],
                                                 'chapter': self.chapters_list[notes_set_number]})


                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}``` in only 1 or 2 sentences..
                    ----------------------------------------
                    Requirtments:
                    0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                    """)
                chain_2 = prompt_2 | self.llm | error_parser
                scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'notes_set': notes_set,
                                          'slide_text': slide_texts[i],
                                          'chapter': self.chapters_list[notes_set_number],
                                          'scripts_temp_1': scripts_temp_1})

            elif i == len(slide_texts) - 2:
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt_1 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please generate a brief script for the summarizing slide: ```{slide_text}```.
                    As a reference, the outline of this lecture is: ```{outline}```.
                    No more than 50 words.
                    ----------------------------------------
                    Requirements:
                    0. Try to be open and inspiring students to think and ask questions.
                    """)
                chain_1 = prompt_1 | self.llm | error_parser
                scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                 'notes_set': notes_set,
                                                 'slide_text': slide_texts[i],
                                                 'outline': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                                 'chapter': self.chapters_list[notes_set_number]})


                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                    Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                    No more than 50 words.
                    ----------------------------------------
                    Requirements:
                    0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                    """)
                chain_2 = prompt_2 | self.llm | error_parser
                scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'notes_set': notes_set,
                                          'slide_text': slide_texts[i],
                                          'outline': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                          'chapter': self.chapters_list[notes_set_number],
                                          'scripts_temp_1': scripts_temp_1})

            elif i != 0 and i != 1 and i != len(slide_texts) - 1 and i != len(slide_texts) - 2:
                if len(slide_texts_temp[i]) < 5:
                    parser = StrOutputParser()
                    error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a script for the slide: ```{slide_text}```.
                        Since the slide is a slide with only a title, please generate a brief script around the title to give an overview with 1 or 2 sentences.
                        As a reference, the content of next slide is: ```{next_slide_text}```.
                        ----------------------------------------
                        Requirements:
                        0. All the information in the slide has been covered.
                        1. The content must be only relevant to the content: ```{slide_text}``` in this specific slide.
                        2. Provide rich examples and explanations and possible applications for the content when needed.
                        3. The response should be directly talking about the academic content, with no introduction or conclusion (like "Today...", or "Now...", "In a word...").
                        """)
                    chain_1 = prompt_1 | self.llm | error_parser
                    scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                     'notes_set': notes_set,
                                                     'slide_text': slide_texts[i],
                                                     'next_slide_text': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                                     'chapter': self.chapters_list[notes_set_number]})


                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                        Since the slide is a slide with only a title, please generate a brief script around the title to give an overview with 1 or 2 sentences.
                        ----------------------------------------
                        Requirements:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        """)
                    chain_2 = prompt_2 | self.llm | error_parser
                    scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                              'notes_set': notes_set,
                                              'slide_text': slide_texts[i],
                                              'next_slide_text': slide_texts[min(i + 1, len(slide_texts) - 1)],
                                              'chapter': self.chapters_list[notes_set_number],
                                              'scripts_temp_1': scripts_temp_1})

                else:
                    parser = StrOutputParser()
                    error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                    prompt_1 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please generate a script for the slide: ```{slide_text}```.
                        Do not talk about the title of this slide. Just focus on the content.
                        Keep in mind that in the previous slide, the basic idea of the concept illustrated in this slide has been introduced.
                        So do not even talk about the definition of this concept. Just focus on the content and explain the content in the slide.
                        ----------------------------------------
                        Requirements:
                        0. All the information in the slide has been covered.
                        1. The content must be only relevant to the content: ```{slide_text}``` in this specific slide.
                        2. Provide rich examples and explanations and possible applications for the content when needed.
                        3. The response should be directly talking about the academic content, with no introduction or conclusion (like "Today...", or "Now...", "In a word...").
                        """)
                    chain_1 = prompt_1 | self.llm | error_parser
                    scripts_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                                     'notes_set': notes_set,
                                                     'slide_text': slide_texts[i],
                                                     'chapter': self.chapters_list[notes_set_number]})


                    prompt_2 = ChatPromptTemplate.from_template(
                        """
                        As a professor teaching chapter: {chapter} in course {zero_shot_topic}.
                        Please refine the script: ```{scripts_temp_1}``` for the slide: ```{slide_text}```.
                        Keep in mind that in the previous slide, the basic idea of the concept illustrated in this slide has been introduced.
                        So do not even talk about the definition of this concept. Just focus on the content and explain the content in the slide.
                        ----------------------------------------
                        Requirements:
                        0. The response should be a fluent colloquial sentences paragraph, from the first word to the last word.
                        1. Remove the first sentence if it is not directly talking about the academic content.
                        """)
                    chain_2 = prompt_2 | self.llm | error_parser
                    scripts = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                              'notes_set': notes_set,
                                              'slide_text': slide_texts[i],
                                              'chapter': self.chapters_list[notes_set_number],
                                              'scripts_temp_1': scripts_temp_1})

            chapter_scripts.append(scripts)
            slides.append(slide_texts[i])
            click.echo(f"Scripts generated for slide {i}")

        file_path = self.videos_dir + f'scripts_for_notes_set{notes_set_number}' + ".json"
        with open(file_path, 'w') as file:
            json.dump(chapter_scripts, file, indent=2)
        click.echo(f"Scripts for note set {notes_set_number} are saved to: {file_path}")
