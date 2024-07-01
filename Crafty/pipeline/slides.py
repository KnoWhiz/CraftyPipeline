import json
import os
import re
import subprocess

import click
import openai

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.pipeline.pipeline_step import PipelineStep
from Crafty.pipeline.utils.network import NetworkUtil
from Crafty.pipeline.utils.tex import TexUtil


class Slides(PipelineStep):

    def __init__(self, para):
        super().__init__(para)

        self.if_short_video = para['if_short_video']
        self.zero_shot_topic = para['topic']
        self.chapters_list = [self.zero_shot_topic]

        os.makedirs(self.videos_dir, exist_ok=True)

        if(self.if_short_video == False):
            self.slides_template_file = para['slides_template_file']
            self.slides_style = para['slides_style']
            self.content_slide_pages = para['content_slide_pages']
            self.chapter = para['chapter']
            self.read_meta_data_from_file()
        elif(self.if_short_video == True):
            self.slides_template_file = para['slides_template_file']
            self.slides_style = para['slides_style']
            self.content_slide_pages = 2
            self.chapter = para['chapter']

    def execute(self):
        if self.chapter is None or self.chapter < 0:
            raise ValueError("Chapter number is not provided or invalid.")

        # Create the full slides for the chapter
        if(self.if_short_video == True):
            full_slides = self.create_full_slides_short(notes_set_number=self.chapter)
        elif(self.if_short_video == False):
            full_slides = self.create_full_slides(notes_set_number=self.chapter)
        click.echo(f'Slides generation finished, next step is generate images.')
        # # Generate images for the slides with only titles
        self.tex_image_generation(full_slides, notes_set_number=self.chapter)
        click.echo(f'Images generation finished, next step is putting images into the Tex file.')
        # # Insert images into TEX file of the slides
        self.insert_images_into_latex(notes_set_number=self.chapter)
        # Compile the slides to PDF
        self.compile_tex_file_to_pdf(notes_set_number=self.chapter)

    def create_full_slides_short(self, notes_set_number=-1):
        """
        Generate short slides.
        """
        slides_template = self._load_slides_template()

        notes_xml = self.notes_dir + f'notes_set{notes_set_number}.xml'
        if os.path.exists(notes_xml):
            with open(notes_xml, 'r') as xml_file:
                notes_set = xml_file.read()
        else:
            raise FileNotFoundError(f"Notes set file not found: {notes_xml}")
        
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        prompt = ChatPromptTemplate.from_template(
            """
            As a short video YouTuber illustrating concept: ```{zero_shot_topic}```.
            Based on the provided material: ```{notes_set}```.
            Please follow the following steps and requirements to generate only {page_number} pages of slides.
            Structure:
            1. First page is the tile with the concept to illustrate
            2. Second page is the explanation of this concept, just a few concise bullet points
            Based on the template in latex format ```{tex_template}``` (But keep in mind that this is only a template, so do not need to include the information in it in your response unless it also shows in the provided material.)
            Important: only response in pure correct latex format. Do not include "```" at the beginning and the end.
            """)
        chain = prompt | self.llm_basic | error_parser
        full_slides = chain.invoke({'zero_shot_topic': self.zero_shot_topic,
                                    'notes_set': notes_set,
                                    'page_number': self.content_slide_pages,
                                    'tex_template': slides_template})
        prompt = ChatPromptTemplate.from_template(
            """
            For course ```{zero_shot_topic}``` and chapter ```{chapter}```.
            Generate a description text for the slides for this lecture within 100 words.
            Start with "This lecture ..." and make sure the generated content is closely tied to the content of the slide.
            Lecture slides:
            ```{full_slides}```
            """)
        chain = prompt | self.llm_basic | error_parser
        video_description = chain.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'chapter': self.chapters_list[notes_set_number],
                                          'full_slides': full_slides})
        with open(self.debug_dir + f"video_description_chapter_{notes_set_number}.json", 'w') as file:
            json.dump(video_description, file, indent=2)

        # Save the response in a .tex file instead of a .txt file
        tex_path = self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex"
        with open(tex_path, 'w') as file:
            file.write(full_slides)
        click.echo(f'Tex file for chapter {notes_set_number} saved to: {tex_path}')
        return full_slides

    def create_full_slides(self, notes_set_number=-1):
        """
        Generate full slides for a given course and chapter based on the provided notes set.
        The slides are generated in LaTeX format and saved to a tex file.
        Calling LLM multiple times to polish the slides.
        """
        slides_template = self._load_slides_template()

        notes_xml = self.notes_dir + f'notes_set{notes_set_number}.xml'
        if os.path.exists(notes_xml):
            with open(notes_xml, 'r') as xml_file:
                notes_set = xml_file.read()
        else:
            raise FileNotFoundError(f"Notes set file not found: {notes_xml}")

        # Send the prompt to the API and get a response
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
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
        chain_1 = prompt_1 | self.llm_advance | error_parser
        full_slides_temp_1 = chain_1.invoke({'zero_shot_topic': self.zero_shot_topic,
                                             'notes_set': notes_set,
                                             'page_number': self.content_slide_pages,
                                             'tex_template': slides_template,
                                             'chapter': self.chapters_list[notes_set_number],
                                             'notes_set_number': notes_set_number})

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
        chain_2 = prompt_2 | self.llm_advance | error_parser
        full_slides_temp_2 = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic,
                                             'notes_set': notes_set,
                                             'page_number': self.content_slide_pages,
                                             'tex_template': slides_template,
                                             'chapter': self.chapters_list[notes_set_number],
                                             'notes_set_number': notes_set_number,
                                             'full_slides_temp_1': full_slides_temp_1})

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
        chain_3 = prompt_3 | self.llm_advance | error_parser
        full_slides_temp_3 = chain_3.invoke({'zero_shot_topic': self.zero_shot_topic,
                                             'notes_set': notes_set,
                                             'page_number': self.content_slide_pages,
                                             'tex_template': slides_template,
                                             'chapter': self.chapters_list[notes_set_number],
                                             'notes_set_number': notes_set_number,
                                             'full_slides_temp_2': full_slides_temp_2})

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
        chain_4 = prompt_4 | self.llm_advance | error_parser
        full_slides = chain_4.invoke({'zero_shot_topic': self.zero_shot_topic,
                                      'notes_set': notes_set,
                                      'page_number': self.content_slide_pages + 2,
                                      'tex_template': slides_template,
                                      'chapter': self.chapters_list[notes_set_number],
                                      'notes_set_number': notes_set_number,
                                      'full_slides_temp_3': full_slides_temp_3})

        prompt = ChatPromptTemplate.from_template(
            """
            For course ```{zero_shot_topic}``` and chapter ```{chapter}```.
            Generate a description text for the slides for this lecture within 100 words.
            Start with "This lecture ..." and make sure the generated content is closely tied to the content of the slide.
            Lecture slides:
            ```{full_slides}```
            """)
        chain = prompt | self.llm_basic | error_parser
        video_description = chain.invoke({'zero_shot_topic': self.zero_shot_topic,
                                          'chapter': self.chapters_list[notes_set_number],
                                          'full_slides': full_slides})
        with open(self.debug_dir + f"video_description_chapter_{notes_set_number}.json", 'w') as file:
            json.dump(video_description, file, indent=2)

        # Save the response in a .tex file instead of a .txt file
        tex_path = self.videos_dir + f'full_slides_for_notes_set{notes_set_number}' + ".tex"
        with open(tex_path, 'w') as file:
            file.write(full_slides)
        click.echo(f'Tex file for chapter {notes_set_number} saved to: {tex_path}')
        return full_slides

    def _load_slides_template(self):
        """
        Loads a LaTeX slides template from a file if specified, or generates a new one based on a given style.
        Returns the LaTeX template content.
        """
        try:
            if self.slides_template_file is None:
                slides_template = TexUtil.generate_latex_template(self.slides_style)
            else:
                slides_template = TexUtil.load_tex_content(self.slides_template_file)
        except Exception as e:
            raise Exception(f"Error loading slides template: {e}")
        return slides_template

    def tex_image_generation(self, full_slides, notes_set_number=-1):
        """
        Generate images for each title slide in the full slides LaTeX file.
        """
        slide_texts_temp = TexUtil.parse_latex_slides(full_slides)
        slide_texts = TexUtil.parse_latex_slides_raw(full_slides)
        for i in range(len(slide_texts)):
            if i >= 2 and slide_texts_temp[i] == '':
                self.generate_dalle_image(prompt=slide_texts[i],
                                          notes_set_number=notes_set_number,
                                          index=i)

    def generate_dalle_image(self, prompt="Crafty", model="dall-e-3", size="1024x1024", quality="standard", notes_set_number=-1, index=0, retry_on_invalid_request=True):
        """
        Generate an image using DALL-E based on a given prompt and save it to a local folder.
        The image is saved with a specific file name based on the notes set number and index.
        """
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_advance)
        prompt_1 = ChatPromptTemplate.from_template(
            """
            For concept: ```{input}``` in course: {zero_shot_topic}, chapter: {chapter}.
            Write a new visual prompt for DALL-E while avoiding any mention of books, signs, titles, text, and words etc.
            Do not include any technical terms, just a simple description.
            Give a graphic description representation of the concept.
            """)
        chain_1 = prompt_1 | self.llm_advance | error_parser
        prompt = chain_1.invoke({'input': prompt, 'zero_shot_topic': self.zero_shot_topic, 'chapter': self.chapters_list[notes_set_number]})

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
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_advance)
                prompt_2 = ChatPromptTemplate.from_template(
                    """
                    For course: {zero_shot_topic}, chapter: {chapter}.
                    Write a new visual prompt for DALL-E while avoiding any mention of books, signs, titles, text, and words etc.
                    Do not include any technical terms, just a simple description.
                    Give a graphic description representation of the concept.
                    Since OpenAI API request was invalid for the previous prompt, try to keep the description safe and harmonious.
                    """)
                chain_2 = prompt_2 | self.llm_advance | error_parser
                prompt = chain_2.invoke({'zero_shot_topic': self.zero_shot_topic, 'chapter': self.chapters_list[notes_set_number]})

                self.generate_dalle_image(prompt=prompt, model=model, size=size, quality=quality, notes_set_number=notes_set_number, index=index, retry_on_invalid_request=False)
            else:
                print(f"Retried with default prompt but encountered an error: {e}")
            return
        except openai.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            return
        # If no exceptions, save the image.
        image_url = response.data[0].url
        NetworkUtil.save_image_from_url(image_url, self.debug_dir, f"chapter_{notes_set_number}_dalle_image_{index}.png")

    def insert_images_into_latex(self, notes_set_number):
        """
        Insert images into the full slides LaTeX file based on the notes set number.
        The images are inserted into frames with minimal content (only titles) to ensure they are displayed correctly.
        """
        latex_file_path = f"{self.videos_dir}full_slides_for_notes_set{notes_set_number}.tex"
        image_file_pattern = rf"chapter_{notes_set_number}_dalle_image_\d+\.png"

        images = [img for img in os.listdir(self.debug_dir) if re.match(image_file_pattern, img)]
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
                # Determine if the frame is "empty" by checking its length or other criteria
                if len(frame_content) <= 2 and image_counter < len(images):  # Adjust criteria as needed
                    # Insert image code before the end frame tag
                    frame_content.insert(-1, \
                                         f"""\\begin{{figure}}[ht]
                        \\centering
                        \\includegraphics[width=0.55\\textwidth]{{{os.path.join(os.path.abspath(self.debug_dir), images[image_counter])}}}
                    \\end{{figure}}\n""")
                    click.echo(f"Inserted image {images[image_counter]} into frame {image_counter + 1}")
                    image_counter += 1
                modified_content.extend(frame_content)
                inside_frame, frame_content = False, []
            elif inside_frame:
                frame_content.append(line)
            else:
                modified_content.append(line)
        with open(latex_file_path, 'w') as file:
            file.writelines(modified_content)
        click.echo(f'Tex file {latex_file_path} updated for images insertion.')

    def compile_tex_file_to_pdf(self, notes_set_number):
        tex_name = f"full_slides_for_notes_set{notes_set_number}.tex"
        latex_file_path = os.path.join(os.path.abspath(self.videos_dir), tex_name)
        # Your command to run xelatex
        command = ['/Library/TeX/texbin/xelatex', latex_file_path]
        # Run subprocess with cwd set to the directory of the .tex file
        with open(self.debug_dir + tex_name + '.log', 'w') as log:
            subprocess.run(command, cwd=os.path.dirname(latex_file_path), stdout=log)
            click.echo(f'PDF file for note set {notes_set_number} saved to: {self.videos_dir}{tex_name.replace(".tex", ".pdf")}')
