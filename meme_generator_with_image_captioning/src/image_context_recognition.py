import os
import requests
from PIL import Image
from hugging_face_model_loader import HuggingFaceModelLoader

class ImageContextRecognition:
    def __init__(self, model_name: str= "llava-hf/llava-1.5-7b-hf", model_quantization: bool=True):
        """
        Initialize the ImageContextRecognition class with a specified model from Hugging Face.
        
        Args:
            model_name (str): The name of the model to be used for image-to-text generation.
        """
        self.model = HuggingFaceModelLoader(
            task="image-to-text", 
            model=model_name,
            model_quantization=model_quantization
        )

    def load_image(self, image_path: str) -> Image:
        """
        Load an image from a local file or a URL.
        
        Args:
            image_path (str): Path to the image file or a URL.
        
        Returns:
            Image: An image object loaded into memory.
        
        Raises:
            ValueError: If the image cannot be loaded from the given path or URL.
        """
        if os.path.exists(image_path):
            return Image.open(image_path)
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()  # Ensures we handle HTTP errors
            return Image.open(response.raw)
        except requests.RequestException as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    def generate_description(self, image_path: str) -> str:
        """
        Generate a description for an image using the loaded model.
        
        Args:
            image_path (str): Path to the image or a URL from which the image will be loaded.
        
        Returns:
            str: A generated text description of the image.
        
        Notes:
            If there is an error in loading the image, the error message is returned as the description.
        """
        try:
            image = self.load_image(image_path)
        except ValueError as e:
            return str(e)  # Returning the error message directly, could also handle differently if needed

        prompt = "USER: <image>\nWrite a description for this image that fully captures both the objects and the scene in the given image\nASSISTANT:"
        result = self.model.generate(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        return result[0]["generated_text"].split("ASSISTANT:")[-1].strip()
