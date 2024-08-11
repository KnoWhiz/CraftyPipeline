import os
import json
import asyncio
from asyncio import Semaphore       # for rate limiting

import xml.etree.ElementTree as ET

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import XMLOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.config import Config, Constants
from Crafty.pipeline.pipeline_step import PipelineStep
from Crafty.pipeline.utils.xml import XmlUtil


class Notes(PipelineStep):
    def __init__(self, para):
        super().__init__(para)

        self.short_video = para['short_video']
        # self.zero_shot_topic = para['topic']
        # self.chapters_list = [self.zero_shot_topic]

        if os.path.exists(self.meta_dir + Config.META_AND_CHAPTERS):
            with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                meta_data = json.load(file)
                if(self.craft_notes != True):
                    self.zero_shot_topic = meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
                else:
                    # Temporary solution for the craft_topic named as zero_shot_topic
                    self.zero_shot_topic = meta_data[Constants.CRAFT_TOPIC_KEY]
        else:
            raise FileNotFoundError(f"Topic file not found in {self.meta_dir}")
        
        if os.path.exists(self.notes_dir + Config.CHAPTERS_AND_SECTIONS):
            with open(self.notes_dir + Config.CHAPTERS_AND_SECTIONS, 'r') as json_file:
                chapters_and_sections = json.load(json_file)
                self.chapters_list = chapters_and_sections[Constants.CHAPTER_LIST_KEY]
                self.sections_list = chapters_and_sections[Constants.SECTION_LIST_KEY]
        else:
            raise FileNotFoundError(f"Chapter and section file not found in {self.notes_dir}")

        print("Chapters list: ", self.chapters_list)
        print("Sections list: ", self.sections_list)

        self.topic = para['topic']
        self.regions = ["Overview", "Examples", "Essentiality"]
        self.max_note_expansion_words = para['max_note_expansion_words']
        self.chapter = para['chapter']
        # Notes will use the model based on user's choice.
        if para['advanced_model']:
            self.llm = self.llm_advance
        else:
            self.llm = self.llm_basic

        if(self.short_video == True):
            self.semaphore = Semaphore(2)       # limit to 2 concurrent executions
        else:
            self.semaphore = Semaphore(1)  
            self.read_meta_data_from_file()

    def execute(self):
        self._find_sections_docs()

        if(self.short_video == True):
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

        else:
            if self.chapter is None or self.chapter < 0:
                raise ValueError("Chapter number is not provided or invalid.")

            chapter_name = self.chapters_list[self.chapter]
            sections = self.sections_list[self.chapter]

            if(self.craft_notes != True):
                notes_exp = self.robust_generate_expansions(chapter_name, sections, 5)
            else:
                notes_exp = self.craft_generate_expansions(self.llm, sections, \
                                                           self.sections_qdocs, \
                                                           self.sections_list[self.chapter], \
                                                           self.zero_shot_topic, \
                                                           self.max_note_expansion_words, \
                                                           self.max_note_expansion_words)

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
    
    def _find_sections_docs(self):
        embed_book = self.main_embedding
        self.sections_qdocs = []
        for i in range(len(self.chapters_list)):
            print("\nSearching qdocs for chapter: ", i)
            sections_temp = self.sections_list[i]
            file_path = os.path.join(self.meta_dir, f'main_qdocs_set{i}.json')
            if not os.path.exists(file_path):
                qdocs_list_temp = []
                for section in sections_temp:
                    docs = embed_book.similarity_search(section, k=4)
                    print("\nDocs for section: ", section)
                    # print("\nDocs: ", docs)
                    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
                    qdocs = qdocs.replace('\u2022', '').replace('\n', '').replace('\no', '').replace('. .', '')
                    qdocs_list_temp.append(qdocs)
                self.sections_qdocs.append(qdocs_list_temp)
                with open(file_path, 'w') as file:
                    json.dump(dict(zip(sections_temp, qdocs_list_temp)), file, indent=2)
            else:
                with open(file_path, 'r') as file:
                    qdocs_list_dict_temp = json.load(file)
                    extracted_qdocs = [qdocs_list_dict_temp[key] for key in sections_temp]
                    self.sections_qdocs.append(extracted_qdocs)

        file_path = os.path.join(self.meta_dir, 'sections_docs.json')
        with open(file_path, 'w') as file:
            json.dump(self.sections_qdocs, file, indent=2)

    def craft_generate_expansions(self, llm, sections, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, max_attempts = 3, regions = ["Outline", "Examples", "Essentiality"]):
        attempt = 0
        while attempt < max_attempts:
            try:
                return asyncio.run(self.craft_generate_expansions_async(llm, sections, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, regions))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating expansions: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate expansions after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Expansions generation failed after {max_attempts} attempts.")

    async def craft_generate_expansions_async(self, llm, sections, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, \
                                        regions = ["Outline", "Examples", "Essentiality"]):
        def format_string(regions):
            markdown_content = "\n".join([f'### {region}\n\nExample content for {region}.\n' for region in regions])
            markdown_format_string = f"""
            {markdown_content}
            """
            return markdown_format_string
        markdown_format_string = format_string(regions)

        inputs = [{
            "max_words_expansion": max_words_expansion,
            "text": text,
            "definition": definition,
            "section": section,
            "course_name_domain": course_name_domain,
            "markdown_format_string": markdown_format_string,
        } for text, section, definition in zip(texts, sections, defs)]
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = ChatPromptTemplate.from_template(
            """
            For the course: {course_name_domain}, provide the expansions with a few pre-defined regions for the section: {section}.
            {section}'s definition is: {definition}.
            
            Generate expansions based on the given context as below:
            Context to extract section definition: {text}.
            Max words for expansion: {max_words_expansion}
            It should formated as markdown:
            {markdown_format_string}

            1. The first region is "Outline" which should be some really brief bullet points about the following content around that sections.
            2. If the concept can be better explained by formulas, use LaTeX syntax in markdown, like:
                ----------------
                $$
                \frac{{a}}{{b}} = \frac{{c}}{{d}}
                $$
                ----------------
            3. If you find you need to add tables, use markdown format, like:
                ----------------
                ### Example Table

                | Header 1   | Header 2   | Header 3   |
                |------------|------------|------------|
                | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
                | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
                | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
                ----------------

            4. Do not include "```markdown" in the response. Final whole response must be in correct markdown format.
            5. Specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
            6. For in-line formulas, use the syntax: $E = mc^2$. Remember must use double ```$``` for display formulas.
            """
        )
        chain = prompt | llm | error_parser
        results = await chain.abatch(inputs)
        return dict(zip(sections, results))
