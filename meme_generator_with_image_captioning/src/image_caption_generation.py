import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import TypedDict, Optional

from hugging_face_model_loader import HuggingFaceModelLoader
# Load environment variables
load_dotenv()

# Define a typed dictionary for meme caption generation
class MemeCaptionGeneration(TypedDict):
    content: str
    
class GeminiMemeCaptionGeneration:
    def __init__(self, model_name: str):
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
    # Function to generate a meme caption based on an image description
    def generate_meme(self, image_description: str) -> Optional[str]:
        """Generate a witty or humorous caption for an image description."""
        prompt = f"""
            You are an expert at creating captions for images. The captions should be humorous or witty, aligning
            with the typical style of social media memes. The caption is delimited by triple backticks.
            Image Description: ```{image_description}```
            """
        try:
            result = self.model.generate_content(
                prompt,
                safety_settings="BLOCK_ONLY_HIGH",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=MemeCaptionGeneration
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
            model=model_name,
            model_quantization=model_quantization
        )

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
        return result