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
def video_generation_params():
    return {
        "if_long_videos": True,             # If true, generate long videos
        "if_short_videos": False,           # Currently do not have short videos
        "script_max_words": 100,            # Currently not used
        "slides_template_file": "3",        # Marking the template file under the folder "templates". User can put their own template file name.
        "slides_style": "simple",           # Only use it if template file is not provided
        "content_slide_pages": 30,          # Number of pages for content slides
        "if_parallel_processing": False,    # If true, use parallel processing (chapters) for video generation
    }

def zero_shot_notes_para(course_description):
    para = {
        "course_info": course_description,                      # Course description
        'llm_source': 'openai',
        'temperature': 0,
        "openai_key_dir": ".env",                               # OpenAI key directory
        "results_dir": "pipeline/test_outputs/",                # Output directory
        "sections_per_chapter": 20,                             # Number of sections per chapter
        "max_note_expansion_words": 500,                        # Maximum number of words for note expansion
        "regions": ["Overview", "Examples", "Essentiality"],    # Regions for note expansion
        "if_advanced_model": False,                             # If true, use advanced model for note expansion (more expensive!)
    }
    para.update(video_generation_params())  # Add video parameters
    return para
```

Then specify learning objective:

```python
if __name__ == "__main__":
    local_test(course_description = "David want to learn about the history of the United States!")
```

Run locally:

```bash
conda activate crafty
cd "<project_dir>"
python local_test.py
```
