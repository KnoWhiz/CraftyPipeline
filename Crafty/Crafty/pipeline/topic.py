import json
import os

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.config import Config
from Crafty.pipeline.pipeline_step import PipelineStep


class Topic(PipelineStep):
    def __init__(self, para):
        super().__init__(para)
        self.meta_dir = Config.OUTPUT_DIR + self.course_id + Config.COURSE_META_DIR
        os.makedirs(self.meta_dir, exist_ok=True)

        self.course_info = para['topic']
        # Topic will use a basic model.
        self.llm = self.llm_basic

    def execute(self):
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
        chain = prompt | self.llm | error_parser
        response = chain.invoke({'course_info': self.course_info})
        with open(self.meta_dir + Config.META_AND_CHAPTERS, 'w') as file:
            json.dump(response, file, indent=2)
        click.echo(f'The meta data is saved in {self.meta_dir + Config.META_AND_CHAPTERS}')
