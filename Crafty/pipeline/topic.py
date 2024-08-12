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
        os.makedirs(self.notes_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

        self.course_info = para['topic']
        # Topic will use a basic model.
        self.llm = self.llm_basic
        self.short_video = para['short_video']

    def execute(self):
        if(self.craft_notes != True):
            response = self.prompt_topic()
        else:
            response = self.craft_topic()
        with open(self.meta_dir + Config.META_AND_CHAPTERS, 'w', encoding='utf-8') as file:
            json.dump(response, file, indent=2, ensure_ascii=False)
        click.echo(f'The meta data is saved in {self.meta_dir + Config.META_AND_CHAPTERS}')

    def prompt_topic(self):
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        if(self.language == 'en'):
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
        elif(self.language == 'zh'):
            prompt = ChatPromptTemplate.from_template(
                """
                需求: \n\n\n
                用中文回答：
                根据学生想要学习的课程信息，这门课程的信息是：```{course_info}```。
                "Context" 是对课程的限制性描述，
                "subject" 是课程的主题，
                "text" 是关于这个用户想要学习的内容的详细描述。
                请回答：这门课程的zero_shot_topic应该是什么，通过结合"context"、"subject"和"text"。
                例如，输入可以是这样的：
                ```
                context: "Bayesian"
                level: "Beginner"
                subject: "Machine learning"
                text: "Bayesian machine learning techniques"
                ```
                回复应该格式化为json：
                ```json
                {{
                "context": <这门课程的上下文是什么>,
                "level": <这门课程的级别是什么>,
                "subject": <这门课程的主题是什么>,
                "zero_shot_topic": <这门课程的zero_shot_topic是什么>
                }}
                ```
                """
            )
        else:
            raise ValueError("Language is not supported.")
        chain = prompt | self.llm | error_parser
        response = chain.invoke({'course_info': self.course_info})
        response['short_video'] = self.short_video
        return response

    def craft_topic(self):
        return self.docs.course_name_domain