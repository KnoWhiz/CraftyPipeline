import json
import os
from abc import ABC, abstractmethod

from Crafty.pipeline.science.api_handler import ApiHandler
from Crafty.pipeline.science.doc_handler import DocHandler
from Crafty.pipeline.science.prompt_handler import PromptHandler
from Crafty.pipeline.utils.hash import HashUtil
from config import Config, Constants


class PipelineStep(ABC):
    def __init__(self, para):
        super().__init__()
        self.llm_basic = ApiHandler(para).models['basic']['instance']
        self.llm_advance = ApiHandler(para).models['advance']['instance']
        self.api = ApiHandler(para)
        self.prompt = PromptHandler(self.api)
        self.llm_basic_context_window = self.api.models['basic']['context_window']
        self.llm_advance_context_window = self.api.models['advance']['context_window']
        self.language = para['language']

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

        # If the user wants to craft the notes
        self.craft_notes = para['craft_notes']
        self.file_name = para['file_name']
        if(self.craft_notes == True):
            self.file_dir = Config.INPUT_DIR
            para['file_dir'] = self.file_dir
            para["results_dir"] = self.meta_dir
            # Create a DocHandler instance, currently only one main file is supported
            para['main_filenames'] = [self.file_name]
            para['supplementary_filenames'] = []
            self.docs = DocHandler(para)
            self.main_embedding = self.docs.main_embedding[0]

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
                if(self.craft_notes != True):
                    self.zero_shot_topic = meta_data[Constants.ZERO_SHOT_TOPIC_KEY]
                else:
                    # Temporary solution for the craft_topic named as zero_shot_topic
                    self.zero_shot_topic = meta_data[Constants.CRAFT_TOPIC_KEY]
        else:
            raise FileNotFoundError(f"Chapter file not found in {self.meta_dir}")