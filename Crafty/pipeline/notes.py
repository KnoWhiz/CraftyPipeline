import asyncio
from asyncio import Semaphore       # for rate limiting

import xml.etree.ElementTree as ET

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.pipeline.pipeline_step import PipelineStep
from Crafty.pipeline.utils.xml import XmlUtil


class Notes(PipelineStep):
    def __init__(self, para):
        super().__init__(para)

        self.if_short_video = para['if_short_video']
        self.zero_shot_topic = para['topic']
        self.chapters_list = [self.zero_shot_topic]

        self.topic = para['topic']
        self.regions = ["Overview", "Examples", "Essentiality"]
        self.max_note_expansion_words = para['max_note_expansion_words']
        self.chapter = para['chapter']
        # Notes will use the model based on user's choice.
        if para['advanced_model']:
            self.llm = self.llm_advance
        else:
            self.llm = self.llm_basic

        if(self.if_short_video != True):
            self.semaphore = Semaphore(1)  
            self.read_meta_data_from_file()
        else:
            self.semaphore = Semaphore(2)       # limit to 2 concurrent executions


    def execute(self):
        if(self.if_short_video == True):
            # Generate notes for short videos
            notes_exp = self.short_generate_expansions(input_prompt=self.topic)

            # Convert notes_exp to XML format
            notes_exp_xml = XmlUtil.dict_to_xml('notes_expansion', notes_exp)

            # Write XML to files
            tree = ET.ElementTree(notes_exp_xml)
            ET.indent(tree)

            note_path = self.notes_dir + 'notes_set0.xml'
            with open(note_path, "wb") as f:
                tree.write(f, encoding="UTF-8", xml_declaration=True)
            click.echo(f'The notes file is saved to: {note_path}')

        elif(self.if_short_video == False):
            if self.chapter is None or self.chapter < 0:
                raise ValueError("Chapter number is not provided or invalid.")

            chapter_name = self.chapters_list[self.chapter]
            sections = self.sections_list[self.chapter]

            notes_exp = self.robust_generate_expansions(chapter_name, sections, 5)

            # Convert notes_exp to XML format
            notes_exp_xml = XmlUtil.dict_to_xml('notes_expansion', notes_exp)

            # Write XML to files
            tree = ET.ElementTree(notes_exp_xml)
            ET.indent(tree)

            note_path = self.notes_dir + f'notes_set{self.chapter}.xml'
            with open(note_path, "wb") as f:
                tree.write(f, encoding="UTF-8", xml_declaration=True)
            click.echo(f'The notes file for chapter {self.chapter} is saved to: {note_path}')

    def short_generate_expansions(self, input_prompt):
        output_instructions = "Provide expansions for the given section in XML format."
        inputs = {
            "input_prompt": input_prompt,
            "output_instructions": output_instructions,
            "expansion_length": self.max_note_expansion_words,
        }

        parser = XMLOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        prompt = ChatPromptTemplate.from_template(
            """
            Your task is to indentify a single key points in the given prompt: {input_prompt} and provide explanations for that point.
            Format the output in XML format as follows:
            ----------------
            {output_instructions}
            ----------------
            Max words for expansion: {expansion_length}
            """
        )
        chain = prompt | self.llm | error_parser
        results = chain.invoke(inputs)
        return dict(zip([input_prompt], [results]))

    def robust_generate_expansions(self, chapter_name, sections, max_attempts=5):
        """
        Generate notes for each section in a robust way, retrying up to a maximum number of attempts in case of failure.

        :param chapter_name: The name of the chapter.
        :param sections: A list of sections for which to generate notes.
        :param max_attempts: The maximum number of attempts to make when generating notes.
        :return: A dictionary mapping section names to generated notes.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                return asyncio.run(
                    self.generate_expansions(chapter_name, sections))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating expansions: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate expansions after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Expansions generation failed after {max_attempts} attempts.")

    async def generate_expansions(self, chapter_name, sections):
        """
        Asynchronously generate notes for each section using the given language model.

        :param chapter_name: The name of the chapter.
        :param sections: A list of sections for which to generate notes.
        :return: A dictionary mapping section names to generated notes.
        """
        async with self.semaphore:    
            inputs = [{
                "course_name": self.zero_shot_topic,
                "chapter_name": chapter_name,
                "section": section,
                "expansion_length": self.max_note_expansion_words,
                "regions": self.regions,
                "output_instructions": XmlUtil.generate_xml_elements(section, self.regions),
            } for section in sections]

            parser = XMLOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
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
        chain = prompt | self.llm | error_parser
        results = await chain.abatch(inputs)

        final_roots = XmlUtil.nest_dict_to_xml(results)

        return dict(zip(sections, final_roots))