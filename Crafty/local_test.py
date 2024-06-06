import json
import sys
import os
from pipeline.dev_tasks import generate_videos

para = {
    # Video generation parameters
    "if_long_videos": True,             # If true, generate long videos
    "if_short_videos": False,           # Currently do not have short videos
    "script_max_words": 100,            # Currently not used
    "slides_template_file": "3",        # Marking the template file under the folder "templates". User can put their own template file name.
    "slides_style": "simple",           # Only use it if template file is not provided
    "content_slide_pages": 30,          # Number of pages for content slides
    "if_parallel_processing": False,    # If true, use parallel processing (chapters) for video generation
    "creative_temperature": 0.5,        # Temperature for creative model

    # Course information
    "course_info": "Hartmut want to learn about the history of the United States!",
    'llm_source': 'openai',
    'temperature': 0,
    "openai_key_dir": ".env",                               # OpenAI key directory
    "results_dir": "pipeline/test_outputs/",                # Output directory
    "sections_per_chapter": 20,                             # Number of sections per chapter
    "max_note_expansion_words": 500,                        # Maximum number of words for note expansion
    "regions": ["Overview", "Examples", "Essentiality"],    # Regions for note expansion
    "if_advanced_model": False,                             # If true, use advanced model for note expansion (more expensive!)
}

if __name__ == "__main__":
    generate_videos(para)