import json
import os

import click
from openai import OpenAI
from pydub import AudioSegment

from Crafty.pipeline.pipeline_step import PipelineStep

# for ChatTTS
import ChatTTS
import torchaudio
import torch
from dotenv import load_dotenv
load_dotenv('sha256.env')
import os
os.environ["PYTORCH_MPS"] = "false"
chat = ChatTTS.Chat()
# download ChatTTS 
chat.load(compile=False)
# chat.load(source='custom', custom_path='/Users/tinad/Desktop/Jun 25 copy/ChatTTS')
class Voice(PipelineStep):

    def __init__(self, para):
        super().__init__(para)

        self.chapter = para['chapter']
        self.read_meta_data_from_file()

    def execute(self):
        if self.chapter is None or self.chapter < 0:
            raise ValueError("Chapter number is not provided or invalid.")

        self.scripts2voice(notes_set_number=self.chapter)

    def scripts2voice(self, speech_file_path=None, input_text=None, model="tts-1", voice="alloy", notes_set_number=-1):
        """
        Converts scripts into mp3 files. If the script files do not exist, it creates all necessary components.
        """
        scripts_file_path = f"{self.videos_dir}scripts_for_notes_set{notes_set_number}.json"

        with open(scripts_file_path, 'r') as json_file:
            scripts = json.load(json_file)  # ["scripts"]

        click.echo(f"Generating voice for {len(scripts)} scripts...")
        for i, script in enumerate(scripts):
            voice_file_path = speech_file_path if speech_file_path and (
                        speech_file_path.endswith("/") and os.path.exists(speech_file_path)) else self.videos_dir
            voice_file_path += f"voice_{i}_chapter_{notes_set_number}.mp3"
        
            spk = torch.load('/Users/tinad/Desktop/Jun 25/seed_1397_restored_emb.pt', map_location=torch.device('cpu'))
           
            params_infer_code = {
                'prompt': '[speed_3]',
                'spk_emb': spk,
            }
            texts = [script]  # Assuming script is a string containing the text to be converted to speech
            
            wavs = chat.infer(texts, params_infer_code=params_infer_code)
            torchaudio.save(voice_file_path, torch.from_numpy(wavs[0]), 24000)

            click.echo(f"Voice {i} saved to: {voice_file_path}")

    def _voice_agent(self, speech_file_path=None, input_text=None, model="tts-1", voice="alloy", notes_set_number=-1):
        """
        Generates an audio speech file from the given text using the specified voice and model, with a 1-second silent time after the content.

        :param speech_file_path: The path where the audio file will be saved. If None, saves to a default directory.
        :param input_text: The text to be converted to speech. If None, a default phrase will be used.
        :param model: The text-to-speech model to use.
        :param voice: The voice model to use for the speech.
        """
        if input_text is None:
            input_text = "input_text not defined"

        if speech_file_path is None:
            speech_file_path = self.videos_dir + f"voice_{-1}_chapter_{notes_set_number}.mp3"

        try:
            # Generate the speech audio
            response = OpenAI().audio.speech.create(model=model, voice=voice, input=input_text)

            # Save the generated speech to a temporary file
            temp_audio_file = f"temp_speech_{notes_set_number}.mp3"
            with open(temp_audio_file, "wb") as f:
                f.write(response.content)

            # Load the speech audio and create a 1-second silence
            speech_audio = AudioSegment.from_file(temp_audio_file)
            one_second_silence = AudioSegment.silent(duration=2000)  # 1,000 milliseconds

            # Combine speech audio with silence
            final_audio = speech_audio + one_second_silence

            # Save the combined audio
            final_audio.export(speech_file_path, format="mp3", parameters=["-ar", "16000"])

            # Clean up the temporary file
            os.remove(temp_audio_file)
        except Exception as e:
            print(f"Failed to generate audio: {e}")

