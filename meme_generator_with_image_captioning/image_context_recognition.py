import requests
from PIL import Image
import os
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline

# load model in 4bit to reduce memory usage & fit in GPU and quantize the weights to float16 instead of float32 to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

class ImageContextRecognition:
    def __init__(self, model_name: str, ):
        self.model_id = model_name
        self.model = pipeline("image-to-text", model=model_name, model_kwargs={"quantization_config": quantization_config})

    def generate_description(self, image_path: str) -> str:
        # load image from local path or url
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            try:
                image = Image.open(requests.get(image_path, stream=True).raw)
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to load image from URL: {e}")
        # generate description
        prompt = "USER: <image>\nWrite a description for this image that fully captures both the objects and the scene in the given image\nASSISTANT:"
        description = self.model(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        return description[0]["generated_text"].split("ASSISTANT:")[-1].strip()