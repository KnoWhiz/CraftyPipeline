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
