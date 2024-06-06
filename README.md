# Crafty
Generate lecture videos with a single prompt!

## Installation

```bash
conda create --name crafty python=3.11
conda activate crafty
pip install -r requirements.txt
```

install MacTex or TeX Live

```bash
# e.g. on macOS or Linux
brew install --cask mactex
```

install ffmpeg

```bash
# e.g. on macOS or Linux
brew install ffmpeg
```

Once installed, you can set the IMAGEIO_FFMPEG_EXE environment variable as indicated in your script. This variable points to the FFmpeg executable, which is typically located in /usr/local/bin/ffmpeg on macOS, but the provided script suggests a Homebrew-specific path under /opt/homebrew/bin/ffmpeg. Verify the correct path using:

```bash
which ffmpeg
```

Then update the environment variable accordingly in your Python script or set it in your shell profile:

```bash
export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
```

## Set OPENAI_API_KEY

```bash
cd "<project_dir>"
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

## Run Native

Edit parameters in local_test.py file:

```python
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
    "course_info": "I want to learn about the history of the United States!",
    'llm_source': 'openai',
    'temperature': 0,
    "openai_key_dir": ".env",                               # OpenAI key directory
    "results_dir": "pipeline/test_outputs/",                # Output directory
    "sections_per_chapter": 20,                             # Number of sections per chapter
    "max_note_expansion_words": 500,                        # Maximum number of words for note expansion
    "regions": ["Overview", "Examples", "Essentiality"],    # Regions for note expansion
    "if_advanced_model": False,                             # If true, use advanced model for note expansion (more expensive!)
}
```

You can specify the learning objective in

```python
"course_info": "I want to learn about the history of the United States!",
```

Run locally:

```bash
conda activate crafty
cd "<project_dir>"
python local_test.py
```

## Work flow

The project is started by running local_test.py. Then "generate_videos" in dev_tasks.py will be called. With the steps in generate_videos, we can create chapters (and sections under them) and then notes (for sections under each chapter). After getting the notes, we can take them as material to run VideoProcessor for videos generation.

### local_test.py
After running local_test.py, we call
```python
generate_videos(para)
```
imported from 
```python
pipeline.dev_tasks
```

### dev_tasks.py
Then we go through the work flow:
1. **Generate chapters:**
    - mynotes.create_chapters()
2. **Generate sections (under chapters):**
    - mynotes.create_sections()
3. **Generate note for each section (regions defined in para):**
    - mynotes.create_notes()
4. **Videos generation for each chapter:**
    - myfulllongvideos.run_parallel_processing() for processing each chapter in parallel using multiprocessing
    - myfulllongvideos.run_sequential_processing() for processing each chapter sequentially
```python
mynotes = Zeroshot_notes(para)
mynotes.create_chapters()
mynotes.create_sections()
mynotes.create_notes()
print(f"Time to create notes: {round((time.time() - st) / 60, 0)} mins of the course for the request {para['course_info'] }.")

para['course_id'] = mynotes.course_id
# print(f"\nCourse ID: {para['course_id']}")

if(para['if_long_videos']):
    myfulllongvideos = VideoProcessor(para)
    if(para['if_parallel_processing']):
        myfulllongvideos.run_parallel_processing()
    else:
        myfulllongvideos.run_sequential_processing()
```

### zeroshot_notes.py
```course_id``` defined by hashing ```self.course_info```. Output files will be saved in ```/pipeline/test_outputs/<material_type>/<course_id>/```.

For notes generation, we properly format learning topic with
```python
_extract_zero_shot_topic(self)
```
in format:
```python
{
"context": <what is the context of this course>,
"level": <what is the level of this course>,
"subject": <what is the subject of this course>,
"zero_shot_topic": <what is the zero_shot_topic of this course>
}
```
Then we go through the process of chapters generation
```python
create_chapters(self)
```
For the list of chapters, generate sections under each chapter in parallel
```python
create_sections(self)
```
The information about chapters and sections will be saved in ```chapters_and_sections.json``` under ```/pipeline/test_outputs/notes/<course_id>/```.

Next by going through each chapter, we generate notes for sections in parallel:
```python
notes_exp = robust_generate_expansions(llm, sections_list_temp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], max_note_expansion_words, 3, self.regions)
```
All files saved as ```notes_set{i}.json```, with ```i``` is the chapter index.

### long_videos.py
After getting all notes, we generate videos will the following steps

1. **Create the full slides for the chapter**
2. **Generate images for the slides with only titles**
    - Currently have a dummy logic generating images for sub-title slides only.
3. **Generate scripts for each slide**
4. **Insert images into TEX file of the slides and compile PDF**
    - Based on MacTex (on Mac)
5. **Generate audio files (.mp3) for the scripts**
    - Could be improved with latest TTS progress: https://bytedancespeech.github.io/seedtts_tech_report/#applications-samples
6. **Convert the full slides PDF to images**
7. **Convert the audio files to MP4 and combine them**
    - Based on ffmpeg and moviepy.editor
    - This is when your computer will start to suffer...

```python
def create_long_videos(self, chapter=0):
    """
    Create long videos for each chapter based on the notes set number.
    """
    # Create the full slides for the chapter
    self.create_full_slides(notes_set_number = chapter)  #"notes_set1"
    # Generate images for the slides with only titles
    self.create_scripts(notes_set_number = chapter)  #"notes_set1"
    # Generate scripts for each slide
    self.tex_image_generation(notes_set_number = chapter)
    # Insert images into TEX file of the slides and compile PDF
    self.insert_images_into_latex(notes_set_number = chapter)
    # Generate audio files for the scripts
    self.scripts2voice(notes_set_number = chapter)
    # Convert the full slides PDF to images
    self.pdf2image(notes_set_number = chapter)
    # Convert the audio files to MP4 and combine them
    self.mp3_to_mp4_and_combine(notes_set_number = chapter)
```

For PDF compiling:
```python
command = ['/Library/TeX/texbin/xelatex', tex_file_path]
subprocess.run(command, cwd=working_directory)
```

## Time consuming and cost

At present, the total time required to generate a script for a chapter video using GPT4 is about 30-40 minutes, and the total time required to generate a script using GPT3.5 is about 10-15 minutes. Among them, the latex generation of ppt takes 2-3 minutes, the script generation of GPT3.5 takes 1-2 minutes, the script generation of GPT4 takes 15-20 minutes, and the voice generation of a 5-6 minute video takes 1-2 minutes. Video synthesis and processing are greatly affected by computer performance and video length, and it is roughly estimated to be about 10-20 minutes. In terms of cost, if GPT4 is used throughout the process to pursue quality, the final video of 16-17 minutes will cost 1.1-1.2 dollars. If GPT3.5 is used for script generation, the video length will be shortened to 5-6 minutes, and the cost will drop to 40-50 cents. If the image generation link is removed, the cost will drop to 30-35 cents. If the voice generation link is removed, the cost will drop to 10-20 cents (mainly from GPT generating slides).
