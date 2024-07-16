import json
import os
import asyncio

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.config import Config, Constants
from Crafty.pipeline.pipeline_step import PipelineStep


class Sections(PipelineStep):
    def __init__(self, para):
        super().__init__(para)

        self.short_video = para['short_video']
        self.zero_shot_topic = para['topic']
        # self.chapters_list = [self.zero_shot_topic]

        self.sections_per_chapter = para['sections_per_chapter']
        # Sections will use an advanced model.
        self.llm = self.llm_advance

    def execute(self):
        if os.path.exists(self.meta_dir + Config.META_AND_CHAPTERS):
            with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                meta_data = json.load(file)
                self.zero_shot_topic = meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
                self.chapters_list = meta_data[Constants.CHAPTERS_KEY]
        else:
            raise FileNotFoundError(f"Chapter file not found in {self.meta_dir}")

        raw_sections_in_chapters = self.robust_generate_sections(self.zero_shot_topic, self.chapters_list)

        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        if(self.short_video == True):
            prompt = ChatPromptTemplate.from_template(
                """
                Based on {raw_sections_in_chapters}, the sections in a list of lists. The length of the list should be the same as the number of chapters.
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
        else:
            prompt = ChatPromptTemplate.from_template(
                """
                Based on {raw_sections_in_chapters}, the sections in a list of lists. The length of the list should be the same as the number of chapters.
                Make sure every section is unique: If one section has a similar meaning with another section in another chapter,
                only keep the first one (with lower chapter index) and remove the other sections.
                Section name should not start with number.
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
        chain = prompt | self.llm | error_parser
        response = chain.invoke({'chapters_list': self.chapters_list, 'raw_sections_in_chapters': raw_sections_in_chapters})
        sections_list = response["sections"]

        with open(self.debug_dir + Config.RAW_SECTIONS_IN_CHAPTER, 'w') as file:
            json.dump(raw_sections_in_chapters, file, indent=2)
        with open(self.notes_dir + Config.CHAPTERS_AND_SECTIONS, 'w') as json_file:
            json.dump({
                Constants.CHAPTER_LIST_KEY: self.chapters_list,
                Constants.SECTION_LIST_KEY: sections_list
        }, json_file, indent=4)
        click.echo(f'The section list is saved with chapter to {self.notes_dir + Config.CHAPTERS_AND_SECTIONS}')

    def robust_generate_sections(self, zero_shot_topic, chapter_list, max_attempts=5):
        """
        Generate sections for each chapter in a robust way, retrying up to a maximum number of attempts in case of failure.

        :param zero_shot_topic: The zero-shot topic of the course.
        :param chapter_list: A list of chapters for which to generate sections.
        :param max_attempts: The maximum number of attempts to make when generating sections.
        :return: A dictionary mapping chapter names to generated sections.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                return asyncio.run(self.generate_sections(zero_shot_topic, chapter_list))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating sections: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate sections after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"sections generation failed after {max_attempts} attempts.")

    async def generate_sections(self, zero_shot_topic, chapter_list):
        """
        Asynchronously generate sections for each chapter using the given language model.

        :param zero_shot_topic: The zero-shot topic of the course.
        :param chapter_list: A list of chapters for which to generate sections.
        :return: A dictionary mapping chapter names to generated sections.
        """
        inputs = [{
            "zero_shot_topic": zero_shot_topic,
            "chapter_name": chapter,
            "sections_per_chapter": self.sections_per_chapter,
        } for chapter in chapter_list]

        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
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
        chain = prompt | self.llm | error_parser
        results = await chain.abatch(inputs)

        return dict(zip(chapter_list, results))
