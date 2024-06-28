import json
import multiprocessing
import os

import click
import fitz
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

from Crafty.pipeline.pipeline_step import PipelineStep


class Video(PipelineStep):

    def __init__(self, para):
        super().__init__(para)
        os.makedirs(self.final_dir, exist_ok=True)
        self.chapter = para['chapter']
        self.read_meta_data_from_file()

    def execute(self):
        if self.chapter is None or self.chapter < 0:
            raise ValueError("Chapter number is not provided or invalid.")

        self.pdf2image(notes_set_number=self.chapter)
        self.mp3_to_mp4_and_combine(notes_set_number=self.chapter)

    def pdf2image(self, notes_set_number=-1):
        """
        Convert the full slides PDF file into images for each page.
        """
        pdf_file_path = self.videos_dir + f"full_slides_for_notes_set{notes_set_number}.pdf"
        doc = fitz.open(pdf_file_path)

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)

            # Increase the dpi by adjusting the zoom factor. Default is 1.0 (72 dpi).
            # For higher resolution, you might use 2.0 (144 dpi) or higher.
            zoom = 8.0  # Adjust this factor to get higher resolution images.
            mat = fitz.Matrix(zoom, zoom)  # The transformation matrix for scaling.

            pix = page.get_pixmap(matrix=mat)  # Use the matrix in get_pixmap
            image_path = self.videos_dir + f"image_{page_number}_chapter_{notes_set_number}.png"
            pix.save(image_path)

    def mp3_to_mp4_and_combine(self, notes_set_number):
        """
        Converts MP3 files into MP4 files using corresponding images as static backgrounds,
        sets a default frame rate (fps) for the video, and combines all MP4 files into one,
        skipping already existing MP4 files and the final combination if it exists, for a specific chapter number.

        :param output_dir: Directory where the MP3 files, PNG files, MP4 files, and the final combined MP4 file are located.
        :param notes_set_number: Specific chapter number to match voice and image files.
        """

        # Define the name of the final combined video file
        final_output_filename = f"combined_video_chapter_{notes_set_number}.mp4"
        final_output_path = os.path.join(self.final_dir, final_output_filename)

        # Check if the combined MP4 file already exists
        if os.path.exists(final_output_path):
            click.echo(f"Combined video {final_output_path} already exists, skipping combination.")
            return  # Exit the function if combined video already exists

        # List all MP3 files and sort them by the index i for the specific chapter
        chapter_str = f"_chapter_{notes_set_number}"
        audio_files = sorted([f for f in os.listdir(self.videos_dir) if f.endswith('.mp3') and chapter_str in f],
                             key=lambda x: int(x.split('_')[1]))

        # List to hold all the individual video clips
        video_clips = []

        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            output_mp4_path = os.path.join(self.videos_dir, f"{base_name}.mp4")

            # Check if MP4 file already exists to avoid re-generating it
            if not os.path.exists(output_mp4_path):
                image_file = f"{base_name.replace('voice_', 'image_')}.png"
                audio_path = os.path.join(self.videos_dir, audio_file)
                image_path = os.path.join(self.videos_dir, image_file)

                if os.path.exists(image_path) and os.path.exists(audio_path):
                    # Load the audio file
                    audio_clip = AudioFileClip(audio_path)

                    # Create an image clip with the same duration as the audio file
                    image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

                    # Set the audio of the image clip as the audio file
                    video_clip = image_clip.set_audio(audio_clip)

                    # Write the individual video clip to a file (MP4)
                    video_clip.write_videofile(output_mp4_path, codec="libx264", audio_codec="aac", fps=12)
                    click.echo(f"Generated {output_mp4_path}")
                else:
                    click.echo(f"Missing files for {base_name}, cannot generate MP4.")
                    continue  # Skip to the next file if either file is missing
            else:
                click.echo(f"MP4 file {output_mp4_path} already exists, skipping generation.")

            # Load the existing or newly created MP4 file for final combination
            video_clips.append(VideoFileClip(output_mp4_path))

        # Combine all the video clips into one video file
        click.echo("Video clips generation done, start to combine.")
        if video_clips:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=12)
            click.echo(f"Generated combined video {final_output_path}")
        else:
            click.echo("No video clips to combine.", err=True)
