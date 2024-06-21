import re


class TexUtil:

    @staticmethod
    def generate_latex_template(style):
        # Pre-defined function to generate LaTeX template, based on the style selected
        base_template = r"""
        \documentclass{beamer}
        \usepackage[utf8]{inputenc}
        \usepackage{graphicx}
        \usepackage{amsmath, amsfonts, amssymb}

        """
        if style == 'simple':
            template = base_template + r"""
            \usetheme{default}
            \begin{document}
            \title{Lecture Title}
            \author{Author Name}
            \date{\today}

            \begin{frame}
            \titlepage
            \end{frame}

            \begin{frame}{Slide Title}
            Content goes here.
            \end{frame}

            \end{document}
            """
        elif style == 'medium':
            template = base_template + r"""
            \usetheme{Madrid}
            \usecolortheme{whale}
            \begin{document}
            \title{Lecture Title}
            \author{Author Name}
            \date{\today}

            \begin{frame}
            \titlepage
            \end{frame}

            \begin{frame}{Slide Title}
            Content goes here.
            \end{frame}

            \end{document}
            """
        elif style == 'complex':
            template = base_template + r"""
            \usetheme{Berlin}
            \useoutertheme{infolines}
            \usecolortheme{orchid}
            \setbeamertemplate{background canvas}[vertical shading][bottom=white,top=blue!10]
            \begin{document}
            \title{Lecture Title}
            \author{Author Name}
            \date{\today}

            \begin{frame}
            \titlepage
            \end{frame}

            \begin{frame}{Slide Title}
            Content goes here.
            \end{frame}

            \end{document}
            """
        else:
            return "Invalid style selected. Please choose 'simple', 'medium', or 'complex'."

        return template

    @staticmethod
    def load_tex_content(file_name):
        file_name = file_name + ".tex"
        # Define the path to the 'templates' folder
        folder_path = 'templates'
        # Construct the full path to the file
        # full_path = f'./{folder_path}/{file_name}'
        full_path = "Crafty/pipeline/templates/" + file_name
        print("\nfull_path for pdf template: ", full_path)
        # Open the file and read its content
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            # If the file is not found, return an informative message
            print("\nFileNotFoundError: ", full_path)
            return f'File {file_name} not found in the templates folder.'
        except Exception as e:
            # For other exceptions, return a message with the error
            return f'An error occurred: {str(e)}'

    @staticmethod
    def parse_latex_slides(latex_content):
        frame_pattern = re.compile(r'\\begin{frame}.*?\\end{frame}', re.DOTALL)
        # Use the pattern to find all occurrences of frame content in the LaTeX document.
        frames = frame_pattern.findall(latex_content)
        # Initialize an empty list to hold the cleaned text of each frame.
        slide_texts = []
        for frame in frames:
            # Remove LaTeX commands within the frame content.
            # This regex matches LaTeX commands, which start with a backslash followed by any number of alphanumeric characters
            # and may include optional arguments in square or curly braces.
            clean_text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*', '', frame)
            # Further clean the extracted text by removing any leftover curly braces and normalizing whitespace.
            # This includes converting multiple spaces, newlines, and tabs into a single space, and trimming leading/trailing spaces.
            clean_text = re.sub(r'[{}]', '', clean_text)  # Remove curly braces
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
            # Append the cleaned text of the current frame to the list of slide texts.
            slide_texts.append(clean_text)
        # Return the list containing the cleaned text of all slides.
        return slide_texts

    @staticmethod
    def parse_latex_slides_raw(latex_content):
        # Functions for parsing LaTeX content while keeping titles
        # Compile a regular expression pattern to identify the content of each slide.
        frame_pattern = re.compile(r'\\begin{frame}(.*?)\\end{frame}', re.DOTALL)
        # Use the pattern to find all occurrences of frame content in the LaTeX document.
        frames = frame_pattern.findall(latex_content)
        # Initialize an empty list to hold the modified text of each frame.
        modified_frames = []
        for frame in frames:
            # Convert all symbols to underscores. A symbol is defined as anything that's not a letter, number, or whitespace.
            modified_frame = re.sub(r'[^\w\s]', '_', frame)
            # Append the modified frame text to the list.
            modified_frames.append(modified_frame)
        # Return the list containing the modified text of all slides.
        return modified_frames
