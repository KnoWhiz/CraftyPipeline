import json
import os

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.config import Config, Constants
from Crafty.pipeline.pipeline_step import PipelineStep


class Chapters(PipelineStep):
    def __init__(self, para):
        super().__init__(para)
        self.meta_dir = Config.OUTPUT_DIR + self.course_id + Config.COURSE_META_DIR

        # Chapters will use an advanced model.
        self.llm = self.llm_advance

    def execute(self):
        if os.path.exists(self.meta_dir + Config.META_AND_CHAPTERS):
            with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                meta_data = json.load(file)
                zero_shot_topic = meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
        else:
            raise FileNotFoundError(f"Meta data file not found in {self.meta_dir}")

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
            "course_name": <course name here>,

            "textbooks": [
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
        chain = prompt | self.llm | error_parser
        response = chain.invoke({'zero_shot_topic': zero_shot_topic})
        meta_data.update(response)
        with open(self.meta_dir + Config.META_AND_CHAPTERS, 'w') as file:
            json.dump(meta_data, file, indent=2)
        click.echo(f'The chapter list is updated into {self.meta_dir + Config.META_AND_CHAPTERS}')
