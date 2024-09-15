import os
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

class ImageContextRecognition:
    def __init__(self, model_name: str):
        """Initialize the model with a specified quantization configuration to reduce memory usage."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = pipeline(
            "image-to-text", 
            model=model_name, 
            model_kwargs={"quantization_config": quantization_config}
        )

    def load_image(self, image_path: str) -> Image:
        """Load an image from a local file or a URL."""
        if os.path.exists(image_path):
            return Image.open(image_path)
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()  # Raises a HTTPError for bad responses
            return Image.open(response.raw)
        except requests.RequestException as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    def generate_description(self, image_path: str) -> str:
        """Generate a description for an image using the loaded model."""
        try:
            image = self.load_image(image_path)
        except ValueError as e:
            return str(e)  # Return error message directly or handle differently as needed

        prompt = "USER: <image>\nWrite a description for this image that fully captures both the objects and the scene in the given image\nASSISTANT:"
        result = self.model(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        return result[0]["generated_text"].split("ASSISTANT:")[-1].strip()