import os
import json
import re
import sys
from dotenv import load_dotenv

from typing import TYPE_CHECKING

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import google.generativeai as genai
from hugging_face_model_loader import HuggingFaceModelLoader
# Load environment variables

# Define a typed dictionary for meme caption generation
class MemeCaptionGeneration(TypedDict):
    content: str
    
    
class GeminiMemeCaptionGeneration:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the generator by configuring it with the API key and setting up the model.
        
        Args:
            model_name (str): Identifier for the model to be loaded from generativeai.
            
        Raises:
            ValueError: If the GEMINI_API_KEY is not found in the environment variables.
        """
        gemini_api_key = self._get_gemini_token()
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def _get_gemini_token(self) -> str:
        """Retrieve the Gemini API token from environment variables."""
        load_dotenv()
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return gemini_api_key
    
    def generate_meme_text(self, image_description: str) -> str:
        """
        Generate a witty or humorous caption for an image description.
        
        Args:
            image_description (str): Description of the image to generate a meme caption for.
        
        Returns:
            Optional[str]: The generated meme text or None if an error occurs.
        """
        prompt = f"""
            You are an expert at creating captions for images. The captions should be humorous or witty, aligning
            with the typical style of social media memes. The caption is delimited by triple backticks.
            Image Description: ```{image_description}```
            """
        try:
            result = self.model.generate_content(
                prompt,
                safety_settings="BLOCK_ONLY_HIGH", #  ensure safety output and block high risk content.
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=MemeCaptionGeneration # return the result in JSON format and the caption is the value of the key "content".
                )
            )
            return result.text
        except Exception as e:
            print(f"Error generating meme: {e}")
            return None

class LLaMa3MemeCaptionGeneration:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", model_quantization: bool = True):
        """
        Initialize the meme caption generator with optional model quantization.
        
        Args:
            model_name (str): Identifier for the model to be loaded from Hugging Face.
            model_quantization (bool): If True, apply quantization to the model to reduce memory usage.
        """
        self.model = HuggingFaceModelLoader(
            task="text-generation", 
            model_name=model_name,
            model_quantization=model_quantization
        ).model

    def generate_meme_text(self, image_description: str) -> str:
        """
        Generate a witty or humorous caption for an image description, formatted for social media memes.
        
        Args:
            image_description (str): Description of the image to generate a meme caption for.
        
        Returns:
            str: A JSON formatted string containing the caption key with the generated text as its value.
        
        """
        prompt = f"""
        You are an expert at creating captions for images. The captions should be humorous or witty, aligning
        with the typical style of social media memes. The caption is delimited by triple backticks.
        Image Description: ```{image_description}```
        
        give the final output in JSON format where key is "caption" and the value is the one generated.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        result = self.model(messages, max_new_tokens=100)
        
        if result[0]['generated_text'] == "":
            raise ValueError("No generated text found")
        
        return json.loads(self._extract_content(result[0]['generated_text'][1]['content']))
    
    def _extract_content(self, text: str) -> str:
        """
        Extracts the content within the first set of curly brackets, including the brackets themselves, 
        from the provided text.

        Args:
            text (str): The string from which to extract content within curly brackets.

        Returns:
            str: The content found within the first set of curly brackets, including the brackets, trimmed of leading 
            and trailing whitespace outside the brackets. If no curly brackets are found, returns an empty string.

        Examples:
            >>> extract_content_with_brackets("Hello {world}!")
            '{world}'

            >>> extract_content_with_brackets("No brackets here")
            ''

            >>> extract_content_with_brackets("{Multi-line content\n goes here} surrounded")
            '{Multi-line content\n goes here}'
    """
    # Regular expression pattern to find text within curly brackets including the brackets
        pattern = r'\{[^}]*\}'

        # Search for the pattern in the text
        match = re.search(pattern, text)

        # Extract and return the content if it exists
        if match:
            return match.group().strip()
        else:
            return ""