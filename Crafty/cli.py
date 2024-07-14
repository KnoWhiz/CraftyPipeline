import click
import os
import sys
import json
# Add the grandparent directory to sys.path, so that Crafty as absolute import can be used.
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, grandparent_dir)

from Crafty.pipeline.chapters import Chapters
from Crafty.pipeline.notes import Notes
from Crafty.pipeline.script import Script
from Crafty.pipeline.sections import Sections
from Crafty.pipeline.slides import Slides
from Crafty.pipeline.topic import Topic
from Crafty.pipeline.video import Video
from Crafty.pipeline.voice import Voice

CONFIG_FILE = "config.json"

def save_config(key, value):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
    else:
        config = {}
    config[key] = value
    
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file)

def load_config(key):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
        return config.get(key)
    return None

@click.group()
def cli():
    pass


@click.command()
@click.option('--topic', help='The learning topic to create items for.', required=True)
@click.option('--llm_source', type=str, help='The source of LLM.', required=False, default='openai')
@click.option('--temperature', type=float, help='The temperature for the basic and advanced model.', required=False, default=0)
@click.option('--creative_temperature', type=float, help='The temperature for the creative model.', required=False, default=0.5)
@click.option('--slides_template_file', type=str, help='The template file for the slides.', required=False)
@click.option('--slides_style', type=str, help='Only use it if template file is not provided.', required=False, default='simple')
@click.option('--content_slide_pages', type=int, help='The number of pages for content slides.', required=False)
@click.option('--parallel_processing', is_flag=True, help='Use parallel processing in the generation.', required=False, default=False)
@click.option('--advanced_model', is_flag=True, help='Use the advanced model for note expansion.', required=False, default=False)
@click.option('--sections_per_chapter', type=int, help='The number of sections per chapter.', required=False, default=20)
@click.option('--max_note_expansion_words', type=int, help='The maximum number of words for note expansion.', required=False, default=500)
@click.option('--if_short_video', is_flag=True, help='Generate short videos instead of full-length videos.', required=False, default=True)
def create(topic, llm_source, temperature, creative_temperature, slides_template_file, slides_style, content_slide_pages, parallel_processing, advanced_model, sections_per_chapter, max_note_expansion_words, if_short_video):
    if content_slide_pages is None:
        content_slide_pages = 2 if if_short_video else 30
    if sections_per_chapter < 5:
        click.echo('Error: sections_per_chapter should be greater or equal to 5.', err=True)
        return
    if slides_template_file is None:
        slides_template_file = '-3' if if_short_video else '3'
    para = {
        'topic': topic,
        'llm_source': llm_source,
        'temperature': temperature,
        'creative_temperature': creative_temperature,
        'slides_template_file': slides_template_file,
        'slides_style': slides_style,
        'content_slide_pages': content_slide_pages,
        'advanced_model': advanced_model,
        'sections_per_chapter': sections_per_chapter,
        'max_note_expansion_words': max_note_expansion_words,
        'if_short_video': if_short_video,
    }
    topic_step = Topic(para)
    click.secho(f'Start generating topic {topic}... Course ID: {topic_step.course_id}', fg='green')
    topic_step.execute()
    para['course_id'] = topic_step.course_id
    click.secho(f'Start generating chapters...', fg='green')
    Chapters(para).execute()
    click.secho(f'Start generating sections...', fg='green')
    section = Sections(para)
    section.execute()
    click.secho(f'Start generating chapters, slides, scripts, voices, videos by chapter...', fg='green')
    for i in range(len(section.chapters_list)):
        # TODO need to implement parallel processing
        para['chapter'] = i
        click.secho(f'Start generating notes for chapter {i}...', fg='green')
        Notes(para).execute()
        click.secho(f'Start generating slides for chapter {i}...', fg='green')
        Slides(para).execute()
        click.secho(f'Start generating scripts for chapter {i}...', fg='green')
        Script(para).execute()
        click.secho(f'Start generating voice for chapter {i}...', fg='green')
        Voice(para).execute()
        click.secho(f'Start generating video for chapter {i}...', fg='green')
        Video(para).execute()
    click.secho('All steps are done.', fg='green')


@click.command()
@click.argument('step', type=click.Choice(['chapter', 'section', 'note', 'slide', 'script', 'voice', 'video']))
@click.option('--topic', help='The learning topic to create items for.', required=False)
@click.option('--course_id', help='The unique ID of the course.', required=False)
@click.option('--llm_source', type=str, help='The source of LLM.', required=False, default='openai')
@click.option('--temperature', type=float, help='The temperature for the basic and advanced model.', required=False, default=0)
@click.option('--creative_temperature', type=float, help='The temperature for the creative model.', required=False, default=0.5)
@click.option('--slides_template_file', type=str, help='The template file for the slides.', required=False)
@click.option('--slides_style', type=str, help='Only use it if template file is not provided.', required=False, default='simple')
@click.option('--content_slide_pages', type=int, help='The number of pages for content slides.', required=False)
@click.option('--advanced_model', is_flag=True, help='Use the advanced model for note expansion.', required=False, default=False)
@click.option('--sections_per_chapter', type=int, help='The number of sections per chapter.', required=False, default=20)
@click.option('--max_note_expansion_words', type=int, help='The maximum number of words for note expansion.', required=False, default=500)
@click.option('--chapter', type=int, help='Only generate output for one chapter.', required=False, default=-1)
@click.option('--if_short_video', is_flag=True, help='Generate short videos instead of full-length videos.', required=False, default=True)
def step(step, topic, course_id, llm_source, temperature, creative_temperature, slides_template_file, slides_style, content_slide_pages, advanced_model, sections_per_chapter, max_note_expansion_words, chapter, if_short_video):
    
    if topic is not None:
        save_config('topic', topic)
    else:
        topic = load_config('topic')
    
    if topic is None:
        click.echo('Error: Please provide a topic.')
        return

    if content_slide_pages is None:
        content_slide_pages = 2 if if_short_video else 30
    if sections_per_chapter < 5:
        click.echo('Error: sections_per_chapter should be greater or equal to 5.', err=True)
        return
    if slides_template_file is None:
        slides_template_file = '-3' if if_short_video else '3'
    para = {
        'topic' : topic,
        'llm_source': llm_source,
        'temperature': temperature,
        'creative_temperature': creative_temperature,
        'slides_template_file': slides_template_file,
        'slides_style': slides_style,
        'content_slide_pages': content_slide_pages,
        'advanced_model': advanced_model,
        'sections_per_chapter': sections_per_chapter,
        'max_note_expansion_words': max_note_expansion_words,
        'chapter': chapter,
        'if_short_video': if_short_video,
    }
    chapter_hint = '' if para['chapter'] == -1 else f' --chapter {para["chapter"]}'

    if course_id is not None:
        para['course_id'] = course_id

    if step == 'chapter':
        if topic is not None:
            para['topic'] = topic
            topic_step = Topic(para)
            para['course_id'] = topic_step.course_id
            click.echo(f'Start generating topic {topic}... Course ID: {para["course_id"]}')
            topic_step.execute()
            chapter_step = Chapters(para)
            click.echo(f'Start generating chapters for Course ID: {para["course_id"]}...')
            chapter_step.execute()
            click.echo('Chapters are generated, please review the file and run next step with:')
            click.secho(f'python Crafty/cli.py step section --course_id {para["course_id"]} --sections_per_chapter 20', fg='green')
        else:
            click.echo('Error: Please provide a topic.')
    elif step == 'section':
        if 'course_id' in para:
            click.echo(f'Generating sections for chapters with course_id {para["course_id"]}...')
            section_step = Sections(para)
            section_step.execute()
            click.echo('Section are generated, please review the file and run next step with:')
            click.secho(f'python Crafty/cli.py step note --course_id {para["course_id"]} --max_note_expansion_words 500 --chapter 0', fg='green')
        else:
            click.echo('Error: Please provide a course_id.')
    elif step == 'note':
        if 'course_id' in para and 'chapter' in para:
            click.echo(f'Generating notes for sections with course_id {para["course_id"]}...')
            if para['advanced_model']:
                para['llm'] = para['llm_advance']
            notes_step = Notes(para)
            notes_step.execute()
            click.echo('Notes file are generated, please review the files and run next step with:')
            click.secho(f'python Crafty/cli.py step slide --course_id {para["course_id"]} --slides_template_file 3 --content_slide_pages 30' + chapter_hint, fg='green')
        else:
            click.echo('Error: Please provide required parameter course_id, chapter.')
    elif step == 'slide':
        if 'course_id' in para and 'chapter' in para:
            click.echo(f'Creating slides for notes with course_id {para["course_id"]}...')
            slides_step = Slides(para)
            slides_step.execute()
            click.echo('Slides files are generated, please review the files and run next step with:')
            click.secho(f'python Crafty/cli.py step script --course_id {para["course_id"]}' + chapter_hint, fg='green')
        else:
            click.echo('Error: Please provide required parameter course_id, chapter.')
    elif step == 'script':
        if 'course_id' in para and 'chapter' in para:
            click.echo(f'Creating scripts for notes with course_id {para["course_id"]}...')
            script_step = Script(para)
            script_step.execute()
            click.echo('Script files are generated, please review the files and run next step with:')
            click.secho(f'python Crafty/cli.py step voice --course_id {para["course_id"]}' + chapter_hint, fg='green')
        else:
            click.echo('Error: Please provide required parameter course_id, chapter.')
    elif step == 'voice':
        if 'course_id' in para and 'chapter' in para:
            click.echo(f'Generating voice for notes with course_id {para["course_id"]}...')
            voice_step = Voice(para)
            voice_step.execute()
            click.echo('Voice files are generated, please review the files and run next step with:')
            click.secho(f'python Crafty/cli.py step video --course_id {para["course_id"]}' + chapter_hint, fg='green')
        else:
            click.echo('Error: Please provide required parameter course_id, chapter.')
    elif step == 'video':
        if 'course_id' in para and 'chapter' in para:
            click.echo(f'Generating video for notes with course_id {para["course_id"]}...')
            video_step = Video(para)
            video_step.execute()
            click.echo('Video files are generated, this is the final step.')
        else:
            click.echo('Error: Please provide required parameter course_id, chapter.')
    else:
        click.echo('Error: Invalid step type.')


cli.add_command(create)
cli.add_command(step)

if __name__ == '__main__':
    cli()