import json
import zipfile
import os
import sqlite3
import pandas as pd
import re

class Page:
    def __init__(self, content, anki_content=None):
        self.page_content = content
        self.metadata = {}
        self.anki_content = anki_content

class AnkiLoader:
    def __init__(self, full_file_path):
        # Full file path
        # full_file_path = "/Users/bingran.you/Downloads/US_Constitutional_Law_Bar_Examination_.apkg"
        self.full_file_path = full_file_path

        # Extract directory path
        directory_path = os.path.dirname(self.full_file_path)
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(self.full_file_path))[0]

        # Create the dictionary
        self.para = {
            "Anki_file_path": directory_path,
            "Anki_file_name": file_name
        }

    def load(self):
        anki_json = Anki2Json(self.para)
        json_data = anki_json.convert_apkg_to_json()
        
        # Transform json_data into a list of Page objects with a 'page_content' attribute
        structured_data = []
        for item in json_data:
            # print("item: ", str(item))
            structured_data.append(Page(str(item), item))

        # print("Structured data loaded successfully: ", (structured_data[0].page_content))
        return structured_data

class Anki2Json:
    def __init__(self, para = None):
        self.file_path = para["Anki_file_path"]
        self.file_name = para["Anki_file_name"]
        self.apkg_path = os.path.join(self.file_path, self.file_name + ".apkg")
        self.json_output_path = os.path.join(self.file_path, self.file_name + ".json")
        self.html_output_path = os.path.join(self.file_path, self.file_name + ".html")

    def _clean_text(self, text):
        if text is None:
            return ""
        clean_text = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', text)
        return clean_text

    def _is_cloze(self, text):
        return bool(re.search(r'\{\{c\d+::.*?\}\}', text))

    def convert_apkg_to_json(self):
        extract_path = self.apkg_path + "_extracted"
        with zipfile.ZipFile(self.apkg_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Rename numbered media files to .jpg
        for root, _, files in os.walk(extract_path):
            for file in files:
                # If the file is numbered, rename it to .jpg
                if file.isdigit():
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.splitext(old_file_path)[0] + '.jpg'
                    os.rename(old_file_path, new_file_path)
                # If the file is media file, rename it to .json
                if file.startswith("media"):
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.splitext(old_file_path)[0] + '.json'
                    os.rename(old_file_path, new_file_path)
        
        db_path = os.path.join(extract_path, 'collection.anki2')
        conn = sqlite3.connect(db_path)

        query = """
        SELECT n.id, flds
        FROM notes n
        """
        
        notes_df = pd.read_sql_query(query, conn)
        notes_df[['Question', 'Answer']] = notes_df['flds'].str.split('\x1f', expand=True)
        notes_df['Question'] = notes_df['Question'].apply(self._clean_text)
        notes_df['Answer'] = notes_df['Answer'].apply(self._clean_text)
        # Remove cloze cards
        notes_df = notes_df[~notes_df['Question'].apply(self._is_cloze)]
        notes_df = notes_df[~notes_df['Answer'].apply(self._is_cloze)]
        # Remove cards with empty Question or Answer fields
        notes_df = notes_df[notes_df['Question'].str.strip() != ""]
        notes_df = notes_df[notes_df['Answer'].str.strip() != ""]
        notes_df = notes_df.drop(columns=['flds'])
        json_data = notes_df.to_dict(orient="records")
        with open(self.json_output_path, "w", encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

        # Load the flashcards and media dictionary
        with open(self.json_output_path, 'r') as file:
            flashcards = json.load(file)
        media_dict_path = os.path.join(extract_path, 'media.json')
        with open(media_dict_path, 'r') as file:
            media_dict = json.load(file)

        # Create a reverse dictionary for media files
        reverse_media_dict = {v: k + '.jpg' for k, v in media_dict.items()}
        print("Reverse media dictionary: ", reverse_media_dict)
        def convert_dict_strings(input_dict):
            converted_dict = {f'"{key}"': f'"{value}"' for key, value in input_dict.items()}
            return converted_dict
        reverse_media_dict = convert_dict_strings(reverse_media_dict)
        path = os.path.join(extract_path, 'reverse_media_dict.json')
        with open(path, 'w') as file:
            json.dump(reverse_media_dict, file, indent=2)

        # Function to replace media links
        def replace_media_links(text):
            for original, replacement in reverse_media_dict.items():
                new_text = text.replace(original, replacement)
                if new_text != text:
                    return new_text
            return text
        # Update the flashcards
        for flashcard in flashcards:
            flashcard['Answer'] = replace_media_links(flashcard['Answer'])
            flashcard['Question'] = replace_media_links(flashcard['Question'])
        # Save the updated flashcards to a new JSON file
        updated_flashcards_path = self.json_output_path
        with open(updated_flashcards_path, 'w') as file:
            json.dump(flashcards, file, indent=2)

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Anki Notes</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h1>Anki Notes</h1>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Question</th>
                    <th>Answer</th>
                </tr>
        """
        for _, row in notes_df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['id']}</td>
                    <td>{row['Question']}</td>
                    <td>{row['Answer']}</td>
                </tr>
            """
        html_content += """
            </table>
        </body>
        </html>
        """

        with open(self.html_output_path, "w", encoding='utf-8') as html_file:
            html_file.write(html_content)

        # # Open the HTML file in the default browser
        # webbrowser.open('file://' + os.path.realpath(self.html_output_path))

        # return self.json_output_path, self.html_output_path
        # return json_data
        return flashcards

# # Example usage
# para = {
#     "Anki_file_path": "/Users/bingran.you/Downloads/",
#     "Anki_file_name": "US_Constitutional_Law_Bar_Examination_"
# }

# anki_json = Anki2Json(para)
# anki_json.convert_apkg_to_json()