## Installation

```bash
conda create --name crafty python=3.11
conda activate crafty
# pip install quart quart-cors langchain openai unstructured pdf2image pdfminer pdfminer.six "langchain[docarray]" tiktoken celery "celery[redis]" gevent eventlet pymongo boto3 scipy chromadb pandas pymupdf langchain_openai langchain_community scikit-learn discord.py
# pip install moviepy pydub
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
