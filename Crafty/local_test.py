import json
import sys
import os
from pipeline.dev_tasks import generate_videos

def video_generation_params():
    return {
        "if_long_videos": False,
        "if_short_videos": False,
        "slide_max_words": 50,
        "script_max_words": 100,
        "slides_template_file": "3",
        "slides_style": "simple",
        "creative_temperature": 0.5,
        "content_slide_pages": 30,
        "if_parallel_processing": False,
    }

def zero_shot_notes_para(course_description):
    para = {
        "course_info": course_description,
        'llm_source': 'openai',
        'temperature': 0,
        "openai_key_dir": ".env",
        "results_dir": "pipeline/test_outputs/",
        "sections_per_chapter": 20,
        "max_note_expansion_words": 500,
        "regions": ["Overview", "Examples", "Essentiality"],
        "if_advanced_model": False,
    }
    para.update(video_generation_params())  # Add video parameters
    return para

def local_test(course_description=None):
    para = zero_shot_notes_para(course_description)
    generate_videos(para)

if __name__ == "__main__":
    local_test(course_description = "David want to learn about the history of the United States!")
