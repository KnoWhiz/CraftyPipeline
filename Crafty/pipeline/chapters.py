import json
import os

import click
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from Crafty.config import Config, Constants
from Crafty.pipeline.pipeline_step import PipelineStep


class Chapters(PipelineStep):
    def __init__(self, para):
        super().__init__(para)

        self.short_video = para['short_video']
        self.zero_shot_topic = para['topic']
        # self.chapters_list = [self.zero_shot_topic]

        self.meta_dir = Config.OUTPUT_DIR + self.course_id + Config.COURSE_META_DIR

        # Chapters will use an advanced model.
        self.llm = self.llm_advance

    def execute(self):
        if(self.craft_notes != True):
            if os.path.exists(self.meta_dir + Config.META_AND_CHAPTERS):
                with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                    self.meta_data = json.load(file)
                    self.zero_shot_topic = self.meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
            else:
                raise FileNotFoundError(f"Meta data file not found in {self.meta_dir}")
            response = self.prompt_chapters()
        else:
            with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                self.meta_data = json.load(file)
                self.craft_topic = self.meta_data[Constants.CRAFT_TOPIC_KEY]
            response = self.craft_chapters()
        self.meta_data.update(response)
        with open(self.meta_dir + Config.META_AND_CHAPTERS, 'w', encoding='utf-8') as file:
            json.dump(self.meta_data, file, indent=2, ensure_ascii=False)
        click.echo(f'The chapter list is updated into {self.meta_dir + Config.META_AND_CHAPTERS}')

    def prompt_chapters(self):
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        if(self.short_video == True):
            if(self.language == 'en'):
                prompt = ChatPromptTemplate.from_template(
                    """
                    Requirements: \n\n\n
                    As as a professor teaching course: {zero_shot_topic}.
                    Please work through the following steps:
                    1. Find one most popular textbooks about this course topic, note down it as ```textbook and author```.
                    2. Based on these textbooks, come up with at most 5 learning sessions that the students can learn the entire course step by step.
                    3. In chapter name, mark the chapter numbers.
                    The output format should be json as follows:
                    ```json
                    {{
                    "course_name": <course name here>,

                    "textbooks": [
                        <textbook here>,
                    ]

                    "authors": [
                        <author here>,
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
            elif(self.language == 'zh'):
                prompt = ChatPromptTemplate.from_template(
                    """
                    需求: \n\n\n
                    用中文回答：
                    作为一位教授教授课程: {zero_shot_topic}.
                    请按照以下步骤进行操作:
                    1. 找到关于这门课程主题的一本最流行的教材，将其记下为```教材和作者```.
                    2. 根据这些教材，提出学生可以逐步学习整个课程的最多5个学习会话。
                    3. 在章节名称中，标记章节编号。
                    输出格式应为以下json:
                    ```json
                    {{
                    "course_name": <课程名称>,

                    "textbooks": [
                        <教科书名称>,
                    ]

                    "authors": [
                        <作者姓名>,
                    ]

                    "Chapters": [
                        <第一章>,
                        <第二章>,
                        ...
                        <第n章>,
                    ]
                    }}
                    ```
                    """)
            else:
                raise ValueError("Language not supported.")
        else:
            if(self.language == 'en'):
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
            elif(self.language == 'zh'):
                prompt = ChatPromptTemplate.from_template(
                    """
                    需求: \n\n\n
                    用中文回答：
                    作为一位教授教授课程: {zero_shot_topic}.
                    请按照以下步骤进行操作:
                    1. 找到关于这门课程主题的3本最流行的教材，将其记下为```教材和作者```.
                    2. 根据这些教材，提出学生可以逐步学习整个课程的最多10个最少5个学习会话。
                    3. 在章节名称中，标记章节编号。
                    输出格式应为以下json:
                    ```json
                    {{
                    "course_name": <课程名称>,

                    "textbooks": [
                        <第一本教科书>,
                        <第二本教科书>,
                        <第三本教科书>,
                    ]

                    "authors": [
                        <第一位作者>,
                        <第二位作者>,
                        <第三位作者>,
                    ]

                    "Chapters": [
                        <第一章>,
                        <第二章>,
                        ...
                        <第n章>,
                    ]
                    }}
                    ```
                    """)
            else:
                raise ValueError("Language not supported.")
        chain = prompt | self.llm | error_parser
        response = chain.invoke({'zero_shot_topic': self.zero_shot_topic})
        return response

    def craft_chapters(self):
        llm = self.llm

        # If the main file type is not a link, generate the chapters using the LLM
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        # error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        if(self.language == 'en'):
            prompt = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                As as a professor teaching course: {course_name_domain}.
                Using textbook with content ```{textbook_content_pages}```.
                Please work through the following steps:
                1. Find the textbook name and author, note down it as ```textbook and author```.
                2. Based on the content attached, find the chapters of this book.
                3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
                The output format should be:
                ```json
                {{
                "Course name": <course name here>,

                "Textbooks": [
                    <textbook here>,
                ]

                "authors": [
                    <author here>,
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
        elif(self.language == 'zh'):
            prompt = ChatPromptTemplate.from_template(
                """
                需求: \n\n\n
                用中文回答：
                作为一位教授教授课程: {course_name_domain}.
                使用初始内容为```{textbook_content_pages}```的教科书。
                请按照以下步骤进行操作:
                1. 找到教科书名称和作者，将其记下为```教科书和作者```.
                2. 根据附加的内容，找到这本书的章节。
                3. 然后按照以下格式记下章节。对于每个章节名称，不要包括章节编号。
                输出格式应为以下json:
                ```json
                {{
                "Course name": <课程名称>,

                "Textbooks": [
                    <教科书>,
                ]

                "authors": [
                    <作者>,
                ]

                "Chapters": [
                    <第一章>,
                    <第二章>,
                    ...
                    <第n章>,
                ]
                }}
                ```
                """)
        else:
            raise ValueError("Language not supported.")
        chain = prompt | self.llm_advance | error_parser
        try:
            response = chain.invoke({'course_name_domain': self.craft_topic, "textbook_content_pages": self.docs.textbook_content_pages})
            # print("\n\nThe response is: ", response)
            self.course_name_textbook_chapters = response
            self.chapters_list = self.course_name_textbook_chapters["Chapters"]
        except Exception as e:
            # Sometimes the API fails to generate the chapters. In such cases, we regenerate the chapters with summarized content.
            chain = prompt | self.llm_basic | error_parser
            textbook_content_summary = self.prompt.summarize_prompt(self.docs.textbook_content_pages, 'basic', custom_token_limit=int(self.llm_basic_context_window/4))
            response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": textbook_content_summary})
            print("\n\nThe course_name_domain response is: ", response)
            self.course_name_textbook_chapters = response
            self.chapters_list = self.course_name_textbook_chapters["Chapters"]

        # Check if the number of chapters is less than 5 and regenerate the chapters if so.
        # print("\nThe list of chapters is: ", self.course_name_textbook_chapters["Chapters"])
        if(len(self.course_name_textbook_chapters["Chapters"]) <= 5 or len(self.course_name_textbook_chapters["Chapters"]) > 15):
            print("\n\nThe number of chapters is less than 5. Please check the chapters.")
            if(self.language == 'en'):
                prompt = ChatPromptTemplate.from_template(
                    """
                    Requirements: \n\n\n
                    As as a professor teaching course: {course_name_domain}.
                    Please work through the following steps:
                    1. Find a textbook name and author for this book, note down it as ```textbook and author```.
                    2. Based on the content attached, find the chapters of this book. The number of chapters should be between 5 and 15.
                    3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
                    The output format should be:
                    ```json
                    {{
                    "Course name": <course name here>,

                    "Textbooks": [
                        <textbook here>,
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
            elif(self.language == 'zh'):
                prompt = ChatPromptTemplate.from_template(
                    """
                    需求: \n\n\n
                    用中文回答：
                    作为一位教授教授课程: {course_name_domain}.
                    请按照以下步骤进行操作:
                    1. 找到这本书的教科书名称和作者，将其记下为```教科书和作者```.
                    2. 根据附加的内容，找到这本书的章节。章节数量应在5到15之间。
                    3. 然后按照以下格式记下章节。对于每个章节名称，不要包括章节编号。
                    输出格式应为以下json:
                    ```json
                    {{
                    "Course name": <课程名称>,

                    "Textbooks": [
                        <教科书>,
                    ]

                    "Chapters": [
                        <第一章>,
                        <第二章>,
                        ...
                        <第n章>,
                    ]
                    }}
                    ```
                    """)
            else:
                raise ValueError("Language not supported.")
            chain = prompt | llm | error_parser
            response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": self.docs.textbook_content_pages})
            self.course_name_textbook_chapters = response
            self.chapters_list = self.course_name_textbook_chapters["Chapters"]

        return response