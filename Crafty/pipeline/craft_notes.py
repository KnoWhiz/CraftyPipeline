import os
import re
import json
import asyncio
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.utils.craft_notes import CardsUtil    # CardsUtil.combine_cards, CardsUtil.find_indices_to_remove, CardsUtil.divide_into_groups, CardsUtil.locate_indices_to_sets

class Craft_notes:
    def __init__(self, para):
        self.options_list = para["options_list"]
        self.rich_content = para["rich_content"]    # True: rich content, False: only use one time prompt
        self.regions = para["regions"]
        self.definition_detail_level = para["definition_detail_level"]
        self.expansion_detail_level = para["expansion_detail_level"]
        self.max_flashcard_definition_words = int(para["definition_detail_level"] * 30 + 20)
        self.max_flashcard_expansion_words = int(para["expansion_detail_level"] * 100 + 100)
        # self.max_flashcard_definition_words = para["max_flashcard_definition_words"]
        # self.max_flashcard_expansion_words = int(para['max_flashcard_expansion_words'])

        self.main_filenames = para.get('main_filenames', [])
        print("\nMain filenames: ", self.main_filenames)
        if len(self.main_filenames) != 1:
            print("Failed. Please only upload one main file.")
            raise Exception("Multiple main file failed")
        self.craft_notes_set_size = int(para['craft_notes_set_size'])
        # self.quality_check_size = int(para['quality_check_size'])
        self.link_craft_notes_size = int(para['link_craft_notes_size'])   # Size of craft_notes generated from links
        self.max_craft_notes_size = int(para['max_craft_notes_size'])

        self.similarity_score_thresh = float(para['similarity_score_thresh'])
        self.keywords_per_page = para["keywords_per_page"]
        #book with no index
        self.num_context_pages = para["num_context_pages"]
        self.page_set_size = para["page_set_size"]
        self.overlapping = para["overlapping"]
        #load llm
        self.api = ApiHandler(para)
        self.llm_advance = self.api.models['advance']['instance']
        self.llm_basic = self.api.models['basic']['instance']
        self.llm_basic_context_window = self.api.models['basic']['context_window']
        self.prompt = PromptHandler(self.api)
        #load documents
        self.docs = DocHandler(para)

        # print("\n The type of docs: ", self.docs.main_file_types)

        self.nMain_doc_pages = self.docs.main_docs_pages[0]
        self.main_doc = self.docs.main_docs[0]
        self.nkeywords = int(min(self.keywords_per_page * self.nMain_doc_pages, self.max_craft_notes_size))
        self.keywords = []
        self.main_embedding = self.docs.main_embedding[0]

    def create_keywords(self):
        """
        Generates keywords from the document content. If a previously generated keywords file exists,
        it loads the keywords from that file. Otherwise, it determines whether the document has an index
        section and selects the appropriate method to generate keywords. Finally, it refines the generated
        keywords.
        """
        if(self.docs.main_file_types[0] != 'apkg' and self.docs.main_file_types[0] != 'link'):
            file_path = os.path.join(self.docs.flashcard_dir,  "sorted_keywords_docs.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    keywords_docs = json.load(file)
                    self.keywords = list(keywords_docs.keys())
                    self.nkeywords = len(self.keywords)
                    self.keywords_qdocs = list(keywords_docs.values())
                return  # Exit the method after loading the keywords

            print("\nCreating keywords...")
            # Check if the document has an index section and call the appropriate method
            if self.docs.indx_page_docs[0].loc[0, 'index_pages'] is not None:
                self._create_keywords_with_index()
            else:
                self._create_keywords_without_index()

            print("\nRefining keywords...")
            self._refine_keywords()
        elif(self.docs.main_file_types[0] == 'apkg'):
            self.keywords = [page.anki_content["Question"] for page in self.docs.main_docs[0]]
            self.nkeywords = len(self.keywords)
            print((self.keywords))
            print("\nLength of keywords: ", self.nkeywords)
        elif(self.docs.main_file_types[0] == 'link'):
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = ChatPromptTemplate.from_template(
                """
                For the given text ```{text}```, please identify a few central keywords that should be emphasized and remembered.

                Output json format:
                ```json
                {{
                "concepts": [
                    <concept 1>,
                    <concept 2>,
                    ...
                    <concept n>
                ]
                }}
                ```
                """)
            chain = prompt | self.llm_advance | error_parser
            self.keywords = chain.invoke({'text': self.docs.textbook_content_pages})["concepts"]
            print("\nKeywords for links: ", self.keywords)
            print("\nLength of keywords: ", len(self.keywords))
            print("\nType of keywords: ", type(self.keywords))
            if(len(self.keywords) > self.link_craft_notes_size):
                # If we have more keywords than the desired number of craft_notes, we need to re-generate the keywords
                print("\nRe-generating keywords for links with smaller amount of keywords...")
                prompt = ChatPromptTemplate.from_template(
                    """
                    For the given text ```{text}```, please identify no more than {nkeys} central keywords that should be emphasized and remembered.

                    Output json format:
                    ```json
                    {{
                    "concepts": [
                        <concept 1>,
                        <concept 2>,
                        ...
                        <concept n>
                    ]
                    }}
                    ```
                    """)
                chain = prompt | self.llm_advance | error_parser
                self.keywords = chain.invoke({'text': self.docs.textbook_content_pages[:self.link_craft_notes_size], 'nkeys': self.link_craft_notes_size})["concepts"]

            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{0}.json')
            with open(file_path, 'w') as file:
                json.dump(self.keywords, file, indent=2)

            # print("\nlink_craft_notes_size: ", self.link_craft_notes_size)
            # print("\nKeywords for links: ", self.keywords)

    def _create_keywords_with_index(self, max_token=2048):
        """
        Creates keywords from the index section of documents using a language model.

        Parameters:
        - max_token (int): The maximum token limit for processing chunks of the index.
        """
        docs = self.docs.indx_page_docs[0]['index_docs']
        index_pages = "".join([docs[i] for i in range(len(docs))])
        index_chunks = self.prompt.split_prompt(str(index_pages), 'basic', custom_token_limit=max_token)
        n = len(index_chunks)
        print(f'index_chunks: {n}.')
        nkeywords_in_chunk = CardsUtil.divide_into_groups(self.nkeywords, ngroup=n)
        print(f'nkeywords: {nkeywords_in_chunk}')
        for i in range(n):
            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{i}.json')
            if not os.path.exists(file_path):
                parser = JsonOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    "course name and its domain: {course_name_domain}\n\n"
                    "As if you were the professor teaching this course, please identify {nkey} critical keywords
                    from the provided index section that are essential for students to understand and memorize.
                    Do not include the explanations of keywords.
                    Do not include keywords that are not central to
                    the course content, such as examples, datasets, exercises, problems, or introductory keywords:
                    "\n\n Index section: {index_docs}"

                    Output json format:
                    ```json
                    {{
                    "Keywords": [
                        <keyword 1>,
                        <keyword 2>,
                        ...
                        <keyword n>
                    ]
                    }}
                    ```
                    """)
                chain = prompt | self.llm_advance | error_parser
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'index_docs': index_chunks[i], 'nkey': nkeywords_in_chunk[i]})

                with open(file_path, 'w') as file:
                    json.dump(response['Keywords'], file, indent=2)

    def _refine_keywords(self):
        if os.path.exists(os.path.join(self.docs.flashcard_dir, 'keywords.json')):
            with open(os.path.join(self.docs.flashcard_dir, 'keywords.json'), 'r') as file:
                self.keywords = json.load(file)
            return
        #refine keywords
        # Create a regex pattern for filenames matching 'craft_notes_set{integer}.json'
        pattern = re.compile(r'^raw_keywords(\d+)\.json$')
        # Use list comprehension to filter files that match the pattern
        raw_keywords_files = [file for file in os.listdir(self.docs.flashcard_dir) if pattern.match(file)]
        raw_keywords = []
        for raw_keywords_file in raw_keywords_files:
            with open(os.path.join(self.docs.flashcard_dir, raw_keywords_file), 'r') as file:
                keywords = json.load(file)
                raw_keywords.append(keywords)
        # self.raw_keywords = " ".join([raw_keywords[i] for i in range(len(raw_keywords))])
        self.raw_keywords = [item for sublist in raw_keywords for item in sublist]

        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        prompt = ChatPromptTemplate.from_template(
            """
            "Given a course and its domain as: {course_name_domain} \n\n"
            "Assume you are a professor teaching this course. Using your knowledge of the general subject
            matter within this domain, please filter out any irrelevant keywords from the given keywords.  Please return the remaining keywords as a list separated by commas. "
            "\n\n keywords: {keywords}"

            Output json format:
            ```json
            {{
            "Keywords": [
                <keyword 1>,
                <keyword 2>,
                ...
                <keyword n>
            ]
            }}
            ```
            """)
        chain = prompt | self.llm_advance | error_parser
        response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'keywords': self.raw_keywords})
        self.keywords = list(set(response['Keywords']))
        self.nkeywords = len(self.keywords)
        print(f"the number of keywords: {len(self.keywords)} ")
        with open(os.path.join(self.docs.flashcard_dir, 'keywords.json'), 'w') as file:
            json.dump(self.keywords, file, indent=2)

    def _create_keywords_without_index(self):
        """
        Generates keywords for a course based on its document content, excluding an index section.
        This process involves segmenting the document, summarizing segments, and extracting essential keywords.
        """
        # Determine the number of chunks by dividing the total page count by the desired chunk size.
        chunk_sizes = CardsUtil.divide_into_groups(self.nMain_doc_pages, group_size=self.num_context_pages)
        nchunks = len(chunk_sizes)
        nkeywords_in_chunk = CardsUtil.divide_into_groups(self.nkeywords, ngroup=nchunks)
        start_page = 0
        # Iterate through each chunk to process its content.
        for i in range(nchunks):
            end_page = start_page + chunk_sizes[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{i}.json')
            if not os.path.exists(file_path):
                temp_page_content = ""
                for k in range(start_page, end_page):
                    temp_page_content += self.main_doc[k].page_content + " "
                ntempkeys = min(self.keywords_per_page * chunk_sizes[i], nkeywords_in_chunk[i]) # The number of temp keys to extract.
                # First step: Use a basic LLM to summarize the chunk's content.
                parser = StrOutputParser()
                prompt = ChatPromptTemplate.from_template(
                    """
                    Requirements: \n\n\n
                    As as a professor teaching the course: {course_name_domain}.
                    From the following text, please summarize every 500 words up to 50 words.
                    text section: {temp_page_content}
                    """)
                chain = prompt | self.llm_basic | parser
                try:
                    response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'temp_page_content': temp_page_content})
                    temp_extracted_content = response
                except Exception as e:
                    # print(f"Failed to summarize the content of chunk {i}: {e}")
                    temp_extracted_content = self.prompt.summarize_prompt(temp_page_content, 'basic', custom_token_limit=int(self.llm_basic_context_window/4))

                # Second step: Use an advanced LLM to identify essential keywords from the summarized content.
                parser = JsonOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    As as a professor teaching course: {course_name_domain}.
                    From the following text, please identify and return only a list of {ntempkeys} critical keywords
                    from the provided word section (many words included) that are essential 
                    for students to understand and memorize.
                    Do not include keywords that are not central it the course content, 
                    such as examples, datasets, exercises, problems, or introductory keywords.
                    text section: {temp_extracted_content}

                    Output json format:
                    ```json
                    {{
                    "Keywords": [
                        <keyword 1>,
                        <keyword 2>,
                        ...
                        <keyword n>
                    ]
                    }}
                    ```
                    """)
                chain = prompt | self.llm_advance | error_parser
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'ntempkeys': ntempkeys, 'temp_extracted_content': temp_extracted_content})
                with open(file_path, 'w') as file:
                    json.dump(response['Keywords'], file, indent=2)
            start_page = end_page

    def _find_keywords_docs(self):
        embed_book = self.main_embedding
        self.keywords_qdocs = []
        for i in range(len(self.chapters_list)):
            print("\nSearching qdocs for chapter: ", i)
            keywords_temp = self.keywords_list[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'main_qdocs_set{i}.json')
            if not os.path.exists(file_path):
                qdocs_list_temp = []
                for keyword in keywords_temp:
                    docs = embed_book.similarity_search(keyword, k=4)
                    print("\nDocs for keyword: ", keyword)
                    # print("\nDocs: ", docs)
                    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
                    qdocs = qdocs.replace('\u2022', '').replace('\n', '').replace('\no', '').replace('. .', '')
                    qdocs_list_temp.append(qdocs)
                    self.keywords_qdocs.append(qdocs)
                with open(file_path, 'w') as file:
                    json.dump(dict(zip(keywords_temp, qdocs_list_temp)), file, indent=2)
            else:
                with open(file_path, 'r') as file:
                    qdocs_list_dict_temp = json.load(file)
                    extracted_qdocs = [qdocs_list_dict_temp[key] for key in keywords_temp]
                    self.keywords_qdocs.extend(extracted_qdocs)

        file_path = os.path.join(self.docs.flashcard_dir, 'keywords_docs.json')
        with open(file_path, 'w') as file:
            json.dump(self.keywords_qdocs, file, indent=2)

    # Definition generation
    async def generate_definitions_async(self, llm, keywords, texts, max_words_craft_notes, max_words_expansion):
        inputs = [{
            "max_words_craft_notes": max_words_craft_notes,
            "text": text,
            "keyword": keyword,
        } for text, keyword in zip(texts, keywords)]
        # parser = JsonOutputParser()
        parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """
            Provide the definition of the keyword: {keyword} in a sentence that is accurate and easy to understand, based on the given context as below:
            Context to extract keyword definition: {text}.
            In the response include no prefix or suffix.
            Max words for definition: {max_words_craft_notes}
            The response should use markdown syntax to highlight important words / parts in bold or underlined,
            but do not include "```markdown" in the response.
            """
        )
        chain = prompt | llm | parser
        results = await chain.abatch(inputs)
        return dict(zip(keywords, results))

    # Definition generation with given number of attempts
    def generate_definitions(self, llm, keywords, texts, max_words_craft_notes, max_words_expansion, max_attempts = 3):
        attempt = 0
        while attempt < max_attempts:
            # return asyncio.run(self.generate_definitions_async(llm, keywords, texts, max_words_craft_notes, max_words_expansion))
            try:
                return asyncio.run(self.generate_definitions_async(llm, keywords, texts, max_words_craft_notes, max_words_expansion))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating definitions: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate definitions after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Definitions generation failed after {max_attempts} attempts.")

    # Expansion generation
    async def generate_expansions_async(self, llm, keywords, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, regions = ["Outline", "Examples", "Essentiality"]):
        def format_string(regions):
            markdown_content = "\n".join([f'### {region}\n\nExample content for {region}.\n' for region in regions])
            markdown_format_string = f"""
            {markdown_content}
            """
            return markdown_format_string
        markdown_format_string = format_string(regions)

        inputs = [{
            "max_words_expansion": max_words_expansion,
            "text": text,
            "definition": definition,
            "keyword": keyword,
            "course_name_domain": course_name_domain,
            "markdown_format_string": markdown_format_string,
        } for text, keyword, definition in zip(texts, keywords, defs)]
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = ChatPromptTemplate.from_template(
            """
            For the course: {course_name_domain}, provide the expansions with a few pre-defined regions for the keyword: {keyword}.
            {keyword}'s definition is: {definition}.
            
            Generate expansions based on the given context as below:
            Context to extract keyword definition: {text}.
            Max words for expansion: {max_words_expansion}
            It should formated as markdown:
            {markdown_format_string}

            Do not include "```markdown" in the response.
            Final whole response must be in correct markdown format.
            And please specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
            """
        )
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """
        #             You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
        #             Knowledge cutoff: 2023-10
        #             Current date: 2024-07-27

        #             Image input capabilities: Enabled
        #             Personality: v2
        #             """
        #         ),
        #         (
        #             "human",
        #             """
        #             For the course: {course_name_domain}, provide the expansions with a few pre-defined regions for the keyword: {keyword}.
        #             {keyword}'s definition is: {definition}.
                    
        #             Generate expansions based on the given context as below:
        #             Context to extract keyword definition: {text}.
        #             Max words for expansion: {max_words_expansion}
        #             It should formated as markdown:
        #             {markdown_format_string}

        #             Do not include "```markdown" in the response.
        #             Final whole response must be in correct markdown format.
        #             And please specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
        #             """
        #         ),
        #     ]
        # )
        chain = prompt | llm | error_parser
        results = await chain.abatch(inputs)
        return dict(zip(keywords, results))

    # Expansion generation with given number of attempts
    def generate_expansions(self, llm, keywords, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, max_attempts = 3, regions = ["Outline", "Examples", "Essentiality"]):
        attempt = 0
        while attempt < max_attempts:
            try:
                return asyncio.run(self.generate_expansions_async(llm, keywords, texts, defs, course_name_domain, max_words_craft_notes, max_words_expansion, regions))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating expansions: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate expansions after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Expansions generation failed after {max_attempts} attempts.")

    # Rich content options generation
    async def generate_rich_content_options(self, llm, keywords, content_list, chapter_name, course_name, options_list = ["Mindmap", "Table", "Formula", "Code", "Image"]):
        """
        Generate rich content format options for the given keywords
        """
        options_map = {
            "Mindmap": """
                Mermaid mindmap in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ```mermaid
                mindmap
                root((Mind Map))
                    subtopic1(Main Topic 1)
                    subsubtopic1(Sub Topic 1.1)
                    subsubtopic2(Sub Topic 1.2)
                        subsubsubtopic1(Sub Sub Topic 1.2.1)
                    subtopic2(Main Topic 2)
                    subsubtopic3(Sub Topic 2.1)
                    subsubtopic4(Sub Topic 2.2)
                    subtopic3(Main Topic 3)
                    subsubtopic5(Sub Topic 3.1)
                    subsubtopic6(Sub Topic 3.2)
                ```
                ----------------
                """,
            "Table": """
                Tables in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ### Example Table

                | Header 1   | Header 2   | Header 3   |
                |------------|------------|------------|
                | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
                | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
                | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
                ----------------
                ```
            """,
            "Formula": """
                Formulas in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ### LaTeX Formulas in Markdown
                This is an inline formula: $E = mc^2$.

                Here is a display formula:
                $$
                \frac{a}{b} = \frac{c}{d}
                $$

                Inline summation formula: $\sum_{i=1}^n i = \frac{n(n+1)}{2}$.
                ----------------
                """,
            "Code": """
                Code Snippets in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ### Example
                Here is a markdown document that includes both inline and block code snippets:

                ```python
                # This function prints a greeting
                def hello_world():
                    print("Hello, World!")

                hello_world()
                ```
                ----------------
                """,
            "Image": "Images in Markdwon"   # No specific format for now
        }
        inputs = [{
            "course_name": course_name,
            "chapter_name": chapter_name,
            "keyword": keyword,
            "content": content_list[keyword],
            "option": options_list,
        } for keyword in keywords]
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = ChatPromptTemplate.from_template(
            """
            In the course: {course_name}, chapter: {chapter_name},
            For keyword: {keyword} what is the most suitable format to illustrate its meaning?
            Answer only one string from the list of options: {option}.
            Do not answer anything other than the options list.
            """
        )
        chain = prompt | llm | error_parser
        options = await chain.abatch(inputs)
        for i in range(len(options)):
            options[i] = re.sub(r'[^a-zA-Z0-9 ]', '', options[i])

        # print("Options: ", options)

        formats = []
        for i in range(len(options)):
            if options[i] in options_list:
                formats.append(options_map[options[i]])
            else:
                formats.append("Sentence")
        
        # print("Formats: ", formats)

        formats = dict(zip(keywords, formats))
        options = dict(zip(keywords, options))
        return formats, options

    # Rich content generation
    async def generate_rich_content(self, llm, keywords, content_list, chapter_name, course_name, formats, options):
        """
        Generate rich content for the given keywords with the given format options
        """
        inputs = [{
            "course_name": course_name,
            "chapter_name": chapter_name,
            "keyword": keyword,
            "content": content_list[keyword],
            "format": formats[keyword],
            "option": options[keyword],
        } for keyword in keywords]
        parser = StrOutputParser()
        # parser = XMLOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = ChatPromptTemplate.from_template(
            """
            For keyword: {keyword}, refine its illustration content: {content}
            by inserting: {option} in the markdown format at the suitable place in the original content
            to make the content more informative.

            Example template for the response:
            {format}

            Important: 
            Refine the rich format content with the real content of the keyword.
            Final whole response must be in correct markdown format. And please specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.

            Do not include the original version in the response. Only respond the refined version.
            """
        )
        chain = prompt | llm | error_parser
        rich_contents = await chain.abatch(inputs)

        # final_roots = nest_dict_to_xml(rich_contents)

        return dict(zip(keywords, rich_contents))

    # Rich content generation with given number of attempts
    def robust_generate_rich_content(self, llm, keywords, content_list, chapter_name, course_name, options_list=["Mindmap", "Table", "Formula", "Code", "Image"], max_attempts = 3, if_parallel = True):
        attempt = 0
        while attempt < max_attempts:
            try:
                if if_parallel:
                    formats, options = asyncio.run(self.generate_rich_content_options(llm, keywords, content_list, chapter_name, course_name, options_list = options_list))
                    return asyncio.run(self.generate_rich_content(llm, keywords, content_list, chapter_name, course_name, formats, options))
                else:
                    results = {}
                    for keyword in keywords:
                        formats, options = asyncio.run(self.generate_rich_content_options(llm, [keyword], content_list, chapter_name, course_name, options_list = options_list))
                        result = asyncio.run(self.generate_rich_content(llm, [keyword], content_list, chapter_name, course_name, formats, options))
                        results.update(result)
                    return results
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for generating rich content: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to generate rich content after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Rich content generation failed after {max_attempts} attempts.")

    def create_craft_notes(self):
        # If the main file type is Link, create craft_notes directly
        if self.docs.main_file_types[0] == 'link':
            print("\nCourse ID is: ", self.docs.course_id)
            print("\nCreating chapters and assigning keywords...")
            self._create_chapters()
            self._asign_keywords_k2c()
            print("\nSearching keywords in documents...")
            self._find_keywords_docs()

        # If the main file type is not Link, proceed with creating craft_notes
        else:
            print("\nCourse ID is: ", self.docs.course_id)
            print("\nCreating chapters and assigning keywords...")
            self._create_chapters()
            try:
                self._asign_keywords_c2k()
            except Exception as e:
                self._asign_keywords_k2c()
            print("\nSearching keywords in documents...")
            self._find_keywords_docs()

        file_path = os.path.join(self.docs.flashcard_dir, "keywords_docs.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.keywords_qdocs = json.load(file)
        else:
            print("keywords_docs.json file not found. Please check the file path.")
        start = 0
        self.full_craft_notes_set = []
        for i in range(len(self.chapters_list)):
            keywords_temp = self.keywords_list[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set{i}.json')
            end = start + len(keywords_temp)
            if not os.path.exists(file_path):
                keywords = []
                texts = []
                # We can further reduce cost here
                for j in range(start, end):
                    keyword = keywords_temp[j-start]
                    keywords.append(keyword)
                    print(f'keyword: {keyword}')

                    qdocs = self.keywords_qdocs[j]
                    # print("\nLength of qdocs: ", len(self.keywords_qdocs))
                    if self.docs.nSupp > 0:
                        qdocs_supps = self._match_docs_in_supp(keyword)
                        # print("\nContent of qdocs_supps: ", qdocs_supps)
                        qdocs = qdocs + qdocs_supps
                    qdocs_summary = self.prompt.summarize_prompt(qdocs, 'basic', custom_token_limit=int(self.llm_basic_context_window/4))
                    texts.append(qdocs_summary)

                file_path = os.path.join(self.docs.flashcard_dir, f'qdocs_set{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(dict(zip(keywords, texts)), file, indent=2)

                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set_def{i}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        cards_def = json.load(file)
                else:
                    cards_def = self.generate_definitions(self.llm_basic, keywords, texts, self.max_flashcard_definition_words, self.max_flashcard_expansion_words)
                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set_def{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(cards_def, file, indent=2)

                definitions_list = [item for item in cards_def.values()]
                keywords = list(cards_def.keys())

                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set_exp{i}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        cards_exp = json.load(file)
                else:
                    try:
                        cards_exp = self.generate_expansions(self.llm_basic, keywords, texts, definitions_list, self.docs.course_name_domain, self.max_flashcard_definition_words, self.max_flashcard_expansion_words, 3, regions=self.regions)
                    except Exception as e:
                        print(f"Error generating expansions for chapter {i}: {e}")
                        # continue  # Skip this iteration and proceed with the next chapter

                chapters_name_temp = self.chapters_list[i]
                keywords_list_temp = self.keywords_list[i]
                
                if(self.rich_content == True):
                    # Generate rich content for the definitions
                    # llm = self.llm_advance
                    print("Generating rich content for the definitions...")
                    llm = self.llm_basic
                    rich_content = self.robust_generate_rich_content(llm, keywords_list_temp, cards_exp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], options_list=self.options_list)
                    cards_exp = rich_content

                craft_notes = CardsUtil.combine_cards(cards_def, cards_exp)
                self.full_craft_notes_set.append(craft_notes)
                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set_exp{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(cards_exp, file, indent=2)
                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(craft_notes, file, indent=2)

            else:
                try:
                    with open(file_path, 'r') as file:
                        craft_notes = json.load(file)
                except json.JSONDecodeError as e:
                    print("JSONDecodeError", e)
                except FileNotFoundError:
                    print("FileNotFoundError: Please check the file path.")
                self.full_craft_notes_set.append(craft_notes)
            start = end
        
        print("Removing duplicated craft_notes...")
        self._remove_duplicated_craft_notes()

    def _match_docs_in_supp(self, keyword):
        qdocs_supps = ""
        for ell in range(self.docs.nSupp):
            docs_supp = self.docs.supp_embedding[ell].similarity_search(keyword, k=4)
            qdocs_supp =  "".join([docs_supp[m].page_content for m in range(len(docs_supp))])
            qdocs_supp = qdocs_supp.replace('\u2022', '').replace('\n', '').replace('\no', '').replace('. .', '')
            qdocs_supps += qdocs_supp
        qdocs_supps_summary = self.prompt.summarize_prompt(qdocs_supps, 'basic', custom_token_limit= int(self.llm_basic_context_window/4))
        return qdocs_supps_summary

    def _remove_duplicated_craft_notes(self):
        # print("\nRemoving duplicated craft_notes...")
        all_craft_notes = {k: v for d in self.full_craft_notes_set for k, v in d.items()}
        keywords = list(all_craft_notes.keys())
        answers = list(all_craft_notes.values())
        # Convert the answers in dict format to JSON strings for similarity comparison
        answers = [json.dumps(d) for d in answers]
        indices = list(CardsUtil.find_indices_to_remove(keywords, texts=[keywords, answers], thresh=self.similarity_score_thresh))

        # Calculate the sets size for each chapter
        self.sets_size = []
        pattern = re.compile(r'^craft_notes_set(\d+)\.json$')
        # List and sort the files based on the number in the file name
        raw_craft_notes_files = sorted(
            [file for file in os.listdir(self.docs.flashcard_dir) if pattern.match(file)],
            key=lambda x: int(pattern.match(x).group(1))
        )
        print("\nFile list: ", raw_craft_notes_files)
        for temp in raw_craft_notes_files:
            with open(os.path.join(self.docs.flashcard_dir, temp), 'r') as file:
                craft_notes_temp = json.load(file)
                self.sets_size.append(len(craft_notes_temp))
        print("\nSets size: ", self.sets_size)
        
        # print(f"Number of duplicates found: {len(indices)}")
        if indices:
            mapped_indices = CardsUtil.locate_indices_to_sets(indices, self.sets_size)
            for i in range(len(indices)):
                set_label = mapped_indices[i][0]
                print("Keywords to remove: " + keywords[indices[i]])
                self.full_craft_notes_set[set_label].pop(keywords[indices[i]], None)
                #Save the modified data back to the file
                file_path = os.path.join(self.docs.flashcard_dir, f'craft_notes_set{set_label}.json')
                with open(file_path, 'w') as file:
                    json.dump(self.full_craft_notes_set[set_label], file, indent=2)
        else:
            print("No duplicates found based on the threshold.")
        return

    def _create_chapters(self):
        llm = self.llm_advance
        path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
        # Check if the course_name_textbook_chapters.json file exists and load the chapters
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.course_name_textbook_chapters = json.load(file)
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]

        # If the file does not exist, generate the chapters
        else:
            # Check if the main file type is a link and generate chapters accordingly
            if(self.docs.main_file_types[0] == 'link'):
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    For the given text ```{text}```, generate a concise title for it within 10 words.
                    """)
                chain = prompt | self.llm_basic | error_parser
                try:
                    response = chain.invoke({'text': self.docs.textbook_content_pages})

                except Exception as e:
                    textbook_content_summary = self.prompt.summarize_prompt(self.docs.textbook_content_pages, 'basic', custom_token_limit=int(self.llm_basic_context_window/4))
                    response = chain.invoke({"text": textbook_content_summary})

                self.chapters_list = [response]
                self.course_name_textbook_chapters = {
                    "Course name": self.docs.course_name_domain,
                    "Textbooks": [self.docs.main_filenames[0]],
                    "Chapters": self.chapters_list
                }
                path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
                with open(path, 'w') as file:
                    json.dump(self.course_name_textbook_chapters, file, indent=2)

                return

            # If the main file type is not a link, generate the chapters using the LLM
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            # error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            prompt = ChatPromptTemplate.from_template(
                """
                Requirements: \n\n\n
                As as a professor teaching course: {course_name_domain}.
                Using textbook with content ```{textbook_content_pages}```.
                Please work through the following steps:
                1. Find the textbook name and author, note down it as ```textbook and author```.
                2. Based on the content attached, find the chapters of this book.
                3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
                The output format should be:
                ```json
                {{
                "Course name": <course name here>,

                "Textbooks": [
                    <textbook here>,
                ]

                "Chapters": [
                    <chapter_1>,
                    <chapter_2>,
                    ...
                    <chapter_n>,
                ]
                }}
                ```
                """)
            chain = prompt | self.llm_advance | error_parser
            try:
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": self.docs.textbook_content_pages})
                # print("\n\nThe response is: ", response)
                self.course_name_textbook_chapters = response
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]
            except Exception as e:
                # Sometimes the API fails to generate the chapters. In such cases, we regenerate the chapters with summarized content.
                chain = prompt | self.llm_basic | error_parser
                textbook_content_summary = self.prompt.summarize_prompt(self.docs.textbook_content_pages, 'basic', custom_token_limit=int(self.llm_basic_context_window/4))
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": textbook_content_summary})
                print("\n\nThe course_name_domain response is: ", response)
                self.course_name_textbook_chapters = response
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]

            # Check if the number of chapters is less than 5 and regenerate the chapters if so.
            # print("\nThe list of chapters is: ", self.course_name_textbook_chapters["Chapters"])
            if(len(self.course_name_textbook_chapters["Chapters"]) <= 5 or len(self.course_name_textbook_chapters["Chapters"]) > 15):
                print("\n\nThe number of chapters is less than 5. Please check the chapters.")
                print("\n\nThe chapters are: ", self.course_name_textbook_chapters["Chapters"])
                prompt = ChatPromptTemplate.from_template(
                    """
                    Requirements: \n\n\n
                    As as a professor teaching course: {course_name_domain}.
                    Please work through the following steps:
                    1. Find a textbook name and author for this book, note down it as ```textbook and author```.
                    2. Based on the content attached, find the chapters of this book. The number of chapters should be between 5 and 15.
                    3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
                    The output format should be:
                    ```json
                    {{
                    "Course name": <course name here>,

                    "Textbooks": [
                        <textbook here>,
                    ]

                    "Chapters": [
                        <chapter_1>,
                        <chapter_2>,
                        ...
                        <chapter_n>,
                    ]
                    }}
                    ```
                    """)
                chain = prompt | llm | error_parser
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": self.docs.textbook_content_pages})
                self.course_name_textbook_chapters = response
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]

            path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
            with open(path, 'w') as file:
                json.dump(self.course_name_textbook_chapters, file, indent=2)

    # Assign keywords to each chapters, from chapters to keywords (c2k) or from keywords to chapters (k2c)
    def _asign_keywords_c2k(self):
        """
        Asigning keywords to each chapter based on the content of the chapters. Review the content of the chapters and assign keywords to each chapter all together.
        Go from chapter to keywords.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        llm = self.llm_advance

        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            print("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.keywords_list = data_temp["keywords_list"]
            return

        self.keywords_min_num = max(int(len(self.keywords) / len(self.chapters_list) / 2), 7)
        self.keywords_max_num = max(int(len(self.keywords) / len(self.chapters_list) / 1.2), 7) + 5
        # print("\nmin_num is: ", self.keywords_min_num)
        # print("max_num is: ", self.keywords_max_num)

        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        # Check if the "keywords_text_in_chapters.txt" file exists. If it does, load and extract the chapters and keywords from the file.
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.keywords_text_in_chapters = file.read()
            print("\n")

            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt_2 = ChatPromptTemplate.from_template(
                """
                Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
                Chapter list: ```{chapters_list}```.
                Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
                Output format in json, with number of lists is equal to the number of chapters:
                ```json
                {{
                "keywords_list": [
                    [<keyword_1>, <keyword_2>, ..., <keyword_n>],
                    [<keyword_1>, <keyword_2>, ..., <keyword_m>],
                    ...
                    [<keyword_1>, <keyword_2>, ..., <keyword_p>],
                ]
                }}
                ```
                """)
            chain_2 = prompt_2 | self.llm_advance | error_parser
            response = chain_2.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        # If the file does not exist, create the chapters and assign keywords to each chapter.
        else:
            print("\n")
            # 3. Make sure each chapter has at least {min_num} keywords and no more than {max_num} keywords.
            parser = StrOutputParser()
            prompt_1 = ChatPromptTemplate.from_template(
                """
                Your task is classifying key concepts of a given course into its given chapter list.
                To solve the problem do the following:
                Things you should know: based on the content of this textbook, it has the following learning chapters
                ```{course_name_textbook_chapters}```
                Based on your own knowledge, for each learning topic (chapters),
                find most relavant keywords that belongs to this chapter from a keywords list: ```{sorted_keywords}```.
                1. Make sure each keyword is assigned to at least one chapter.
                2. Make sure each chapter has at least {min_num} keywords and no more than {max_num} keywords.
                3. Do not have similar keywords in different chapters.
                Use the following format:
                Course name:
                <course name here>
                Learning topics:
                <chapter here>
                <Keywords: keywords for the above topic here>
                """)
            chain_1 = prompt_1 | self.llm_advance | parser
            response = chain_1.invoke({'course_name_textbook_chapters': self.course_name_textbook_chapters, 'sorted_keywords': self.keywords, 'min_num': self.keywords_min_num, 'max_num': self.keywords_max_num})
            self.keywords_text_in_chapters = response
            
            print("\n")
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt_3 = ChatPromptTemplate.from_template(
                """
                Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
                Chapter list: ```{chapters_list}```.
                Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
                Output format in json, with number of lists is equal to the number of chapters:
                ```json
                {{
                "keywords_list": [
                    [<keyword_1>, <keyword_2>, ..., <keyword_n>],
                    [<keyword_1>, <keyword_2>, ..., <keyword_m>],
                    ...
                    [<keyword_1>, <keyword_2>, ..., <keyword_p>],
                ]
                }}
                ```
                """)
            chain_3 = prompt_3 | self.llm_basic | error_parser
            response = chain_3.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        # If the keywords in some chapters are less than the minimum number of keywords, reassign keywords to the chapters.
        if(any(len(keyword_group) < 5 for keyword_group in self.keywords_list)):
            print("\nThe number of keywords in some chapters is less than 5. Reassigning keywords to the chapters...")
            parser = StrOutputParser()
            prompt_1 = ChatPromptTemplate.from_template(
                """
                Your task is classifying key concepts of a given course into its given chapter list.
                To solve the problem do the following:
                Things you should know: based on the content of this textbook, it has the following learning chapters
                ```{course_name_textbook_chapters}```
                And the keywords list for the list of chapters is ```{keywords_list_original}```.

                Refine tha chapters list, and refine the number of keywords in each chapter by picking and using keywords from a large keywords list: ```{sorted_keywords}```. Add more keywords to each chapter if the number of keywords is less than {min_num}.
                Make sure each chapter has at least {min_num} keywords.
                Use the following format:
                Course name:
                <course name here>
                Learning topics:
                <chapter here>
                <Keywords: keywords for the above topic here>
                """)
            chain_1 = prompt_1 | self.llm_advance | parser
            response = chain_1.invoke({'course_name_textbook_chapters': self.course_name_textbook_chapters, 'sorted_keywords': self.keywords, 'min_num': self.keywords_min_num, 'max_num': self.keywords_max_num, 'keywords_list_original': self.keywords_list})
            self.keywords_text_in_chapters = response

            print("\n")
            parser = JsonOutputParser()
            prompt_3 = ChatPromptTemplate.from_template(
                """
                Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
                Chapter list: ```{chapters_list}```.
                Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
                Output format in json, with number of lists is equal to the number of chapters:
                ```json
                {{
                "keywords_list": [
                    [<keyword_1>, <keyword_2>, ..., <keyword_n>],
                    [<keyword_1>, <keyword_2>, ..., <keyword_m>],
                    ...
                    [<keyword_1>, <keyword_2>, ..., <keyword_p>],
                ]
                }}
                ```
                """)
            chain_3 = prompt_3 | self.llm_basic | parser
            response = chain_3.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        with open(path, 'w') as file:
            file.write(self.keywords_text_in_chapters)
        path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        with open(path, 'w') as file:
            json.dump(self.chapters_list, file, indent=2)
        path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        with open(path, 'w') as file:
            json.dump(self.keywords_list, file, indent=2)

        print("\n\nself.chapters_list are:\n\n", self.chapters_list)
        print("\n\nself.keywords_list are:\n\n", self.keywords_list)

        data_temp = {
            "chapters_list": self.chapters_list,
            "keywords_list": self.keywords_list
        }

        if(len(self.chapters_list) != len(self.keywords_list)):
            raise ValueError("The number of chapters and keywords do not match.")

        # Save to JSON file
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        with open(path, 'w') as json_file:
            json.dump(data_temp, json_file, indent=4)
        return data_temp

    # Chapter assignment generation
    async def assign_chapters_async(self, llm, keywords, chapters, course_name):
        inputs = [{
                "course_name": course_name,
                "chapters": chapters,
                "keyword": keyword,
                } for keyword in keywords]
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = ChatPromptTemplate.from_template(
            """
            Course name: {course_name}
            Chapter list: {chapters}
            Which chapter does the keyword '{keyword}' belong to? Use the same chapter name as in the chapter list.
            response in the json format:
            ```json
            {{
            "chapter": <chapter name here>
            }}
            ```
            """
        )
        chain = prompt | llm | error_parser
        results = await chain.abatch(inputs)
        return dict(zip(keywords, results))

    # Chapter assignment generation with given number of attempts
    def assign_chapters(self, llm, keywords, chapters, course_name, max_attempts = 3):
        attempt = 0
        while attempt < max_attempts:
            try:
                data = asyncio.run(self.assign_chapters_async(llm, keywords, chapters, course_name))
                ordered_chapters = chapters

                # Initialize an empty dictionary
                chapters_dict = {}

                # Populate the dictionary with chapters and their respective keywords
                for keyword, info in data.items():
                    chapter = info['chapter']
                    if chapter not in chapters_dict:
                        chapters_dict[chapter] = []
                    chapters_dict[chapter].append(keyword)

                # # Filter and sort the chapters_dict based on the ordered_chapters list
                # sorted_chapters_dict = {chapter: chapters_dict[chapter] for chapter in ordered_chapters if chapter in chapters_dict}

                # Extract chapter names and their keywords
                chapter_names = list(chapters_dict.keys())
                chapter_keywords = [' '.join(keywords) for keywords in chapters_dict.values()]
                # Create a CountVectorizer instance
                vectorizer = CountVectorizer().fit_transform(ordered_chapters + chapter_names)
                vectors = vectorizer.toarray()
                # Calculate cosine similarity
                cosine_matrix = cosine_similarity(vectors)
                # Find the best match for each chapter in ordered_chapters
                sorted_chapters_dict = {}
                for idx, ordered_chapter in enumerate(ordered_chapters):
                    # Get the cosine similarity scores for the current chapter
                    similarities = cosine_matrix[idx, len(ordered_chapters):]
                    # Find the index of the best match
                    best_match_idx = similarities.argmax()
                    best_match_chapter = chapter_names[best_match_idx]
                    sorted_chapters_dict[ordered_chapter] = chapters_dict[best_match_chapter]

                # Extract chapters and keywords into separate lists
                chapters_list = list(sorted_chapters_dict.keys())
                keywords_list = list(sorted_chapters_dict.values())

                # Construct the final JSON structure
                final_data = {
                    'chapters_list': chapters_list,
                    'keywords_list': keywords_list
                }
                return final_data
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for assigning chapters: {e}")
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to assign chapters after {max_attempts} attempts.")
                    # Return None or raise an exception depending on how you want to handle complete failure.
                    raise Exception(f"Chapter assignment failed after {max_attempts} attempts.")

    # Assign keywords to each chapters, from chapters to keywords (c2k) or from keywords to chapters (k2c)
    def _asign_keywords_k2c(self):
        """
        Asigning each keyword an index of chapter given the list of chapters.
        Go from keywords to chapters.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        llm = self.llm_basic
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            print("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
            self.chapters_list = data_temp["chapters_list"]
            self.keywords_list = data_temp["keywords_list"]
            self.keywords = [keyword for keyword_group in self.keywords_list for keyword in keyword_group]
            self.nkeywords = len(self.keywords)
            print(f"The number of final keywords from loaded file: {self.nkeywords}")
        else:
            # Send the prompt to the API and get response
            data_temp = self.assign_chapters(llm = self.llm_basic, keywords = self.keywords, chapters = self.chapters_list, course_name = self.docs.course_name_domain)
            self.chapters_list = data_temp["chapters_list"]
            self.keywords_list = data_temp["keywords_list"]
            self.keywords = [keyword for keyword_group in self.keywords_list for keyword in keyword_group]
            self.nkeywords = len(self.keywords)
            print(f"The number of final keywords: {self.nkeywords}")
            # Save to JSON file
            with open(path, 'w') as json_file:
                json.dump(data_temp, json_file, indent=4)

    def get_chapters_craft_notes_list(self):
        return self.full_craft_notes_set

    def get_all_craft_notes_list(self):
        all_craft_notes = {k: v for d in self.full_craft_notes_set for k, v in d.items()}
        return all_craft_notes

    def get_chapters_list(self):
        return self.chapters_list

    def get_hash_id(self):
        return self.docs.course_id

    def get_course_name(self):
        if "Course name" in self.course_name_textbook_chapters:
            return self.course_name_textbook_chapters["Course name"]
        else:
            return ""