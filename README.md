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

Once installed, you can set the ```IMAGEIO_FFMPEG_EXE``` environment variable as indicated in your script. This variable points to the FFmpeg executable, which is typically located in ```/usr/local/bin/ffmpeg``` on macOS, but the provided script suggests a Homebrew-specific path under ```/opt/homebrew/bin/ffmpeg```. Verify the correct path using:

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

The project is started by running ```local_test.py```. Then ```generate_videos``` in ```dev_tasks.py``` will be called. With the steps in ```generate_videos```, we can create chapters (and sections under them) and then notes (for sections under each chapter). After getting the notes, we can take them as material to run ```VideoProcessor``` for videos generation.

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
All files saved as ```notes_set{i}.json``` under ```/pipeline/test_outputs/notes/<course_id>/```, with ```i``` is the chapter index.

Examples:
```
<?xml version='1.0' encoding='UTF-8'?>
<notes_expansion>
  <root>
    <Introduction_to_PreColumbian_America>
      <item>
        <Overview>Pre-Columbian America refers to the time period before the arrival of Christopher Columbus in 1492. This era is characterized by the diverse cultures and civilizations that existed in the Americas, including the Aztec, Maya, and Inca civilizations. These societies had complex social structures, advanced agricultural practices, and unique artistic and architectural achievements.</Overview>
      </item>
      <item>
        <Examples>- The Aztec Empire, located in present-day Mexico, was known for its capital city of Tenochtitlan, which was built on an island in the middle of Lake Texcoco. The Aztecs were skilled engineers and built impressive temples and pyramids.
        - The Maya civilization, located in present-day Mexico and Central America, developed a sophisticated writing system, calendar, and mathematical system. They also built impressive cities with elaborate stone structures.
        - The Inca Empire, located in present-day Peru, was the largest empire in pre-Columbian America. The Inca were known for their advanced agricultural techniques, such as terraced farming, and their extensive road network.</Examples>
      </item>
      <item>
        <Essentiality>Understanding pre-Columbian America is essential for gaining a comprehensive view of the history of the United States. The interactions between Native American societies and European explorers and colonizers had a profound impact on the development of the Americas. By studying pre-Columbian America, we can gain insights into the rich cultural heritage of the indigenous peoples of the Americas and the complex dynamics of early encounters between different civilizations.</Essentiality>
      </item>
    </Introduction_to_PreColumbian_America>
    <Geography_and_Climate_of_North_America>
      <item>
        <Overview>North America is a vast continent with diverse geography and climate. It is bordered by the Arctic Ocean to the north, the Atlantic Ocean to the east, the Pacific Ocean to the west, and the Gulf of Mexico to the south. The continent is home to a wide range of landscapes, including mountains, plains, deserts, and forests.</Overview>
      </item>
      <item>
        <Examples>- The Rocky Mountains run along the western part of North America, stretching from Canada down to the southwestern United States.
        - The Great Plains cover much of the central part of the continent, providing fertile land for agriculture.
        - The Amazon Rainforest in South America is the largest tropical rainforest in the world, playing a crucial role in the planet's ecosystem.
        - The Great Lakes, located in the northern part of the continent, are a series of interconnected freshwater lakes that are important for transportation and recreation.</Examples>
      </item>
      <item>
        <Essentiality>Understanding the geography and climate of North America is essential for understanding the history and development of the continent. The diverse landscapes and climates have influenced the way people have lived and interacted with their environment throughout history. From the early Native American civilizations to the European explorers and settlers, geography and climate have played a significant role in shaping the course of events in North America.</Essentiality>
      </item>
    </Geography_and_Climate_of_North_America>
    <Indigenous_Cultures_and_Societies>
      <item>
        <Overview>Indigenous cultures and societies in pre-Columbian America were diverse and complex, with a wide range of languages, customs, and traditions. These societies had developed sophisticated agricultural practices, architectural achievements, and social structures long before the arrival of Europeans.</Overview>
      </item>
      <item>
        <Examples>Some examples of indigenous cultures and societies in North America include the Aztecs, Mayans, Incas, Iroquois Confederacy, and Pueblo people. Each of these groups had unique cultural practices, such as the Aztecs' complex calendar system and ritual sacrifices, the Mayans' advanced knowledge of astronomy and mathematics, and the Iroquois Confederacy's system of government based on consensus and cooperation.</Examples>
      </item>
      <item>
        <Essentiality>Understanding indigenous cultures and societies is essential for gaining a comprehensive view of the history of the United States. These societies played a crucial role in shaping the land, resources, and social structures that European colonizers encountered upon their arrival. Recognizing the diversity and complexity of indigenous cultures also helps to challenge stereotypes and misconceptions that have persisted throughout history.</Essentiality>
      </item>
    </Indigenous_Cultures_and_Societies>
    <Major_PreColumbian_Civilizations_The_Maya_Aztec_and_Inca>
      <item>
        <Overview>The Maya, Aztec, and Inca were three major pre-Columbian civilizations that thrived in the Americas before the arrival of Europeans. Each civilization had its own unique characteristics and achievements that contributed to the rich tapestry of indigenous cultures in the region.</Overview>
      </item>
      <item>
        <Examples>The Maya civilization, located in present-day Mexico and Central America, is known for its advanced knowledge of astronomy, mathematics, and writing system. They built impressive cities with elaborate stone temples and palaces, such as Tikal and Chichen Itza.

        The Aztec civilization, centered in present-day Mexico City, was known for its powerful empire and sophisticated social structure. They built the city of Tenochtitlan on an island in Lake Texcoco, which became one of the largest cities in the world at the time.

        The Inca civilization, located in the Andes Mountains of South America, is famous for its engineering marvels, such as the stone city of Machu Picchu. They also had a highly organized society with a complex system of roads and communication networks.</Examples>
      </item>
      <item>
        <Essentiality>These civilizations were essential in shaping the cultural and historical landscape of the Americas. They developed complex societies with advanced technologies and artistic achievements that continue to fascinate scholars and visitors alike. The Maya, Aztec, and Inca left a lasting legacy that is still celebrated and studied today.</Essentiality>
      </item>
```

### long_videos.py
After getting all notes, we generate videos will the following steps

1. **Create the full slides for the chapter**
2. **Generate images for the slides with only titles**
    - Currently have a dummy logic generating images for sub-title slides only.
3. **Generate scripts for each slide**
4. **Insert images into TEX file of the slides and compile PDF**
    - Based on MacTex (on Mac)
5. **Generate audio files (.mp3) for the scripts**
    - Could be improved with latest TTS progress in the future: https://bytedancespeech.github.io/seedtts_tech_report/#applications-samples
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
