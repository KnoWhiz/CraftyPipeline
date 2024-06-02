import json
import sys
import os
from pipeline.dev_tasks import generate_notes

def video_generation_params():
    return {
        "if_long_videos": True,
        "if_short_videos": False,
        "slide_max_words": 50,
        "script_max_words": 100,
        "slides_template_file": "3",
        "slides_style": "simple",
        "creative_temperature": 0.5,
        "content_slide_pages": 30,
        "chapter_number": 2,
    }

def zero_shot_notes_para(course_description):
    para = {
        "is_zero_shot": True,
        "course_info": course_description,
        'llm_source': 'openai',
        'temperature': 0,
        "openai_key_dir": ".env",
        "results_dir": "pipeline/test_outputs/",
        "keywords_per_chapter": 20,
        "notes_set_size": 30,
        "quality_check_size": 50,
        "max_notes_size": 300,
        "max_definition_length": 50,
        "max_note_expansion_words": 200,
        "max_quiz_questions_per_section": 10,
        "quiz_random_seed": 5,
        "max_test_multiple_choice_questions_per_section": 1,
        "max_test_short_answer_questions_per_section": 1,
    }
    para.update(video_generation_params())  # Add video parameters
    return para

def notes_para(main_filenames, supplementary_filenames=None):
    # filename_without_ext = filename
    para = {
        "is_zero_shot": False,
        "book_dir": "pipeline/test_inputs/",
        'llm_source': 'openai',
        'temperature': 0,
        "openai_key_dir": ".env",
        "main_filenames": main_filenames,
        "supplementary_filenames": supplementary_filenames,
        "results_dir": "pipeline/test_outputs/",
        "course_id_mapping_file": "pipeline/test_outputs/course_id_mapping.json",
        "chunk_size": 2000,
        "notes_set_size": 30,
        "quality_check_size": 30,

        "similarity_score_thresh": 0.8,
        "max_notes_size": 300,
        "max_note_definition_words": 50,
        "max_note_expansion_words": 200,
        "num_context_pages": 15,
        "keywords_per_page": 1.5,
        "page_set_size": 5,
        "overlapping": 0,
        "if_filter": False,
        "max_note_expansion_words": 200,
        "max_quiz_questions_per_section": 10,
        "quiz_random_seed": 5,
        "max_test_multiple_choice_questions_per_section": 1,
        "max_test_short_answer_questions_per_section": 1,
    }
    para.update(video_generation_params())  # Add video parameters
    return para

def local_test(is_zero_shot, course_description=None, main_files=None, supplementary_files=None):
    if is_zero_shot:
        para = zero_shot_notes_para(course_description)
    else:
        para = notes_para(main_files, supplementary_files)
    generate_notes(para)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("python local_test.py filename")

    local_test(is_zero_shot=False, main_files = ["GeneralBiology.pdf"], supplementary_files = [])
    # local_test(is_zero_shot=False, main_files = ["1.apkg"],supplementary_files = [])
    # local_test(is_zero_shot=False, main_files = ["10.pdf"],
    #            supplementary_files = [])
    # local_test(is_zero_shot=False, main_files = ["9.pdf"], supplementary_files = [])
    # local_test(is_zero_shot=False, main_files = ["0.pdf"], supplementary_files = [])
    # local_test(is_zero_shot=False, main_files = ["9.pdf"],
    #            supplementary_files = [])
    # local_test(is_zero_shot=True, course_description = "want to learn EM physics.", learner_info=learner_info)
    # local_test(is_zero_shot=True, course_description = "want to learn EM physics..", learner_info=learner_info)
    # local_test(is_zero_shot=True, course_description = "I want to learn Japanese history...")
    # local_test(is_zero_shot=True, course_description = "want to learn data visualization technique.")
    # local_test(is_zero_shot=True, course_description = "Joe want to learn college level statistics...")
    # local_test(is_zero_shot=True, course_description = "Alice want to deeply learn quantum mechanics...")
    # local_test(is_zero_shot=True, course_description = "Bob: want to learn web development with React.")
    # local_test(is_zero_shot=True, course_description = "I want to learn about the history of the United States.")
    # local_test(is_zero_shot=True, course_description = "I want to learn about political science..")
    # local_test(is_zero_shot=True, course_description = "I want to learn about psychology.")
    # Already have full content, 9c0b754218ac5cc1a3d324501e358f64a90355039f8fe9b1ccbf6b1e
    # local_test(is_zero_shot=True, course_description = "I want to learn about Criminal Law, college level.")
    # Only have half content, e7fff2e3bb0237e02e8c0cec1a94dcad1c51e829c44df363d23b0d74
    # local_test(is_zero_shot=True, course_description = "I want to learn a Contracts Crash Course for business school.")
    # Run overnight, 5760571873f72707079f91b969e7b393eaa5ca9a6490ed8ec9015fad
    # local_test(is_zero_shot=True, course_description = "I want to learn about Property Crash Course for law school.")
    # Run, 8ea6991e90dc158b10e7ecd7f18b28f2d0885e646ef076b420944e9d
    # local_test(is_zero_shot=True, course_description = "I want to learn about Constitutional Law Crash Course for law school.")
    # Run on Win, d3631a300297a1dc4b7606917ea48dc266fb933e5a371bf86498bfe4
    # local_test(is_zero_shot=True, course_description="I want to learn about Civil Procedure Crash Course for law school.")
    # f22a65114d62bc444983e395fa01d25244361d18299e52a0c0bd56f6
    # local_test(is_zero_shot=True, course_description = "I want to learn about Business Entrepreneurship Crash Course for MBA.")
    # fe548397c8632896bf65a4982161e15f63da1385c0c8592bc8bc2f9d
    # local_test(is_zero_shot=True, course_description = "I want to learn about Business Management Crash Course for Business School.")
    # d1bef43ff0a76c784dcc9dd749ddd4375c34694e09bbeabbc3c71f6f
    # local_test(is_zero_shot=True, course_description = "I want to learn about Leadership & Management Crash Course for Business School.")
    # 3f400c775da4b82c4cf335410e47daaf6f46bbe2178ef2f9f517d5d0
    # local_test(is_zero_shot=True, course_description = "I want to learn about How To Write Business Plan for Business School.")
    #
    # local_test(is_zero_shot=True, course_description = "How can you to become a successful entrepreneur.")
    # 1747311a9c951ef0d0c349c413fcab08fdb5f67f86dab760b8abf025
    # local_test(is_zero_shot=True, course_description = "I want to learn Evidence Crash Course in Law School.")
    #
    # local_test(is_zero_shot=True, course_description = "I want to learn about the history of the United States test... I have no idea just want to learn..")
