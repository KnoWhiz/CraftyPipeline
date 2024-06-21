import json
import os
from abc import ABC, abstractmethod

from Crafty.pipeline.science.api_handler import ApiHandler
from Crafty.pipeline.utils.hash import HashUtil
from config import Config, Constants


class PipelineStep(ABC):
    def __init__(self, para):
        super().__init__()
        self.llm_basic = ApiHandler(para).models['basic']['instance']
        self.llm_advance = ApiHandler(para).models['advance']['instance']

        if 'course_id' in para:
            self.course_id = para['course_id']
        else:
            if 'topic' in para:
                self.course_id = HashUtil.course_id(para['topic'])
                self.course_info = para["topic"]
            else:
                raise ValueError("Topic and course ID are missing in the parameters.")

        self.meta_dir = Config.OUTPUT_DIR + self.course_id + Config.COURSE_META_DIR
        self.notes_dir = Config.OUTPUT_DIR + self.course_id + Config.NOTES_DIR
        self.debug_dir = Config.OUTPUT_DIR + self.course_id + Config.DEBUG_DIR
        self.videos_dir = Config.OUTPUT_DIR + self.course_id + Config.VIDEOS_DIR
        self.final_dir = Config.OUTPUT_DIR + self.course_id + Config.FINAL_DIR

    @abstractmethod
    def execute(self):
        pass

    def read_meta_data_from_file(self):
        if os.path.exists(self.notes_dir + Config.CHAPTERS_AND_SECTIONS):
            with open(self.notes_dir + Config.CHAPTERS_AND_SECTIONS, 'r') as json_file:
                chapters_and_sections = json.load(json_file)
                self.chapters_list = chapters_and_sections[Constants.CHAPTER_LIST_KEY]
                self.sections_list = chapters_and_sections[Constants.SECTION_LIST_KEY]
        else:
            raise FileNotFoundError(f"Chapter and section file not found in {self.notes_dir}")

        if os.path.exists(self.meta_dir + Config.META_AND_CHAPTERS):
            with open(self.meta_dir + Config.META_AND_CHAPTERS, 'r') as file:
                meta_data = json.load(file)
                self.zero_shot_topic = meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
        else:
            raise FileNotFoundError(f"Chapter file not found in {self.meta_dir}")