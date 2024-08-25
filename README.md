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

install openlimit

The latest version of openlimit is not available on pypi, and the version on pypi doesn't compatible with Python 3.11+

If you need to install the latest version. Execute the following commands anywhere in your workspace to install openlimit.

```bash
git clone https://github.com/shobrook/openlimit.git
cd openlimit
pip install .
```

## Set OPENAI_API_KEY

```bash
cd "<project_dir>"
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

## CLI Commands

Crafty provides two main commands: `create` and `step`.

### Create

The `create` command is used to create a new course video from given topic. To use the `create` command, simply type:

```bash
python Crafty/cli.py create --topic "I would like to learn about ..."
```

The `create` command has several optional parameters that can be used to customize the behavior of the command. Here's a brief description of each:

- `--temperature <float>`: This parameter sets the temperature for the basic and advanced model. The default value is 0.

- `--creative_temperature <float>`: This parameter sets the temperature for the creative model. The default value is 0.5.

- `--slides_template_file <str>`: This parameter specifies the template file to use for generating slides. The default value is '3'.

- `--slides_style <str>`: This parameter specifies the style of the slides. It should only be used if a template file is not provided. The default value is 'simple'.

- `--content_slide_pages <int>`: This parameter sets the number of pages to generate for content slides. The default value is 30.

- `--parallel_processing`: This flag indicates whether to use parallel processing in the generation. It does not require a value. If used, it sets the value to True.

- `--advanced_model`: This flag indicates whether to use the advanced model for note expansion. It does not require a value. If used, it sets the value to True.

- `--sections_per_chapter <int>`: This parameter sets the number of sections per chapter. The default value is 20.

- `--max_note_expansion_words <int>`: This parameter sets the maximum number of words for note expansion. The default value is 500.

- `--short_video`: This flag indicates whether to generate short videos.

- `--language <str>`: This parameter sets the perfered language for all content generation. The default language is English ("en"), and also supports Chinese ("zh").

These parameters can be used as follows:

```bash
python Crafty/cli.py create --topic <topic> --temperature <float> --creative_temperature <float> --slides_template_file <str> --slides_style <str> --content_slide_pages <int> --parallel_processing --advanced_model --sections_per_chapter <int> --max_note_expansion_words <int>
```

```bash
python Crafty/cli.py create --topic "I want to learn EM physics" --temperature 0.0 --creative_temperature 0.3 --slides_template_file "1"  --content_slide_pages 10 --parallel_processing --advanced_model --sections_per_chapter 10 --max_note_expansion_words 100 --language "zh"
```

Replace `<topic>`, `<float>`, `<str>`, and `<int>` with the actual values you want to use. If you want to use the `--parallel_processing` or `--advanced_model` flags, simply include them in the command without a value.

### Step

The `step` command is used to execute a specific step in the course creation process. The steps should be executed in the following order:

1. `chapter`
1. `section`
1. `note`
1. `slide`
1. `script`
1. `voice`
1. `video`

Here's how to use each step:

#### Chapter

You should always start with chapter command to create meta data and chapters for a given learning topic.

```bash
python Crafty/cli.py step chapter --topic <topic>
```

After the first step, each step will prompt you the next step to execute in the console. Please follow the instructions to continue.

#### Section

Start from second step, you are going to provide the course_id instead of topic to continue using existing materials.

```bash
python Crafty/cli.py step section --course_id <course_id> --sections_per_chapter 20
```

`--sections_per_chapter` is the number of sections you want to create for each chapter. The default value is 20.

#### Note

To generate notes for the sections of a course, use the `note` step. Starting from notes step, you must use `--chapter` to specify which chapter you want to generate.

```bash
python Crafty/cli.py step note --course_id <course_id> --max_note_expansion_words 500 --chapter 0
```

`--max_note_expansion_words` is the maximum number of words to expand the notes. The default value is 500.

`--chapter` is the chapter index to generate notes for. The chapter number start from 0.

Here is an example of notes generation for a course with 3 chapters:

You can revise the notes before proceeding to the next step.

```xml
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
    </Major_PreColumbian_Civilizations_The_Maya_Aztec_and_Inca>
  </root>
</notes_expansion>

```

#### Slide

To create slides for the notes of a course, use the `slide` step.

```bash
python Crafty/cli.py step slide --course_id <course_id> --slides_template_file 3 --content_slide_pages 30 --chapter 0
```

`--slides_template_file` is the template file to use for generating slides. The default value is 3.

`--content_slide_pages` is the number of pages to generate for content slides. The default value is 30.

#### Script

To create scripts for the slides of a course, use the `script` step.

```bash
python Crafty/cli.py step script --course_id <course_id> --chapter 0
```

#### Voice

To generate voice for reading the scripts of a course, use the `voice` step.

```bash
python Crafty/cli.py step voice --course_id <course_id> --chapter 0
```

#### Video

Finally, to create videos from the voices and slides of a course, use the `video` step.

```bash
python Crafty/cli.py step video --course_id <course_id> --chapter 0
```

## Time consuming and cost

At present, the total time required to generate a script for a chapter video using GPT4 is about 30-40 minutes, and the total time required to generate a script using GPT3.5 is about 10-15 minutes. Among them, the latex generation of ppt takes 2-3 minutes, the script generation of GPT3.5 takes 1-2 minutes, the script generation of GPT4 takes 15-20 minutes, and the voice generation of a 5-6 minute video takes 1-2 minutes. Video synthesis and processing are greatly affected by computer performance and video length, and it is roughly estimated to be about 10-20 minutes. In terms of cost, if GPT4 is used throughout the process to pursue quality, the final video of 16-17 minutes will cost 1.1-1.2 dollars. If GPT3.5 is used for script generation, the video length will be shortened to 5-6 minutes, and the cost will drop to 40-50 cents. If the image generation link is removed, the cost will drop to 30-35 cents. If the voice generation link is removed, the cost will drop to 10-20 cents (mainly from GPT generating slides).
