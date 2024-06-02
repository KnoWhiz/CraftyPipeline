from pipeline.science.notes import notes
from pipeline.science.zeroshot_notes import Zeroshot_notes

from pipeline.science.long_videos import Long_videos
from pipeline.science.short_videos import Short_videos
from pipeline.science.long_videos import VideoProcessor

import time

def generate_videos(para):
	st = time.time()
	if para["is_zero_shot"]:
		mynotes = Zeroshot_notes(para)
		mynotes.create_chapters()
		mynotes.create_keywords()
		mynotes.create_notes()
		print(f"Time to create notes: {round((time.time() - st) / 60, 0)} mins of the course for the request {para['course_info'] }.")

		para['course_id'] = mynotes.course_id
		# print(f"\nCourse ID: {para['course_id']}")

		if(para['if_long_videos']):
			myfulllongvideos = VideoProcessor(para)
			myfulllongvideos.run_parallel_processing()
			# myfulllongvideos.run_sequential_processing()
		
		# if(para['if_short_videos']):
		# 	myshortvideos = Short_videos(para)
	else:
		mynotes = notes(para)
		mynotes.create_keywords()

		# mynotes.create_chapters()
		# mynotes.asign_keywords()

		mynotes.create_notes()
		print(f"Time to create notes: {round((time.time() - st) / 60, 0)} mins for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

		para['course_id'] = mynotes.docs.course_id
		# print(f"\nCourse ID: {para['course_id']}")

		if(para['if_long_videos']):
			myfulllongvideos = VideoProcessor(para)
			myfulllongvideos.run_parallel_processing()
			# myfulllongvideos.run_sequential_processing()
			
		# if(para['if_short_videos']):
		# 	myshortvideos = Short_videos(para)