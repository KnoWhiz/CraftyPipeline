import os
import sys
import time
import asyncio
import random
import re
import json
import pandas as pd
import numpy as np
import shutil
import scipy.stats as stats
from typing import List, Optional, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler

class notes:
    def __init__(self, para_lectures):
        pass