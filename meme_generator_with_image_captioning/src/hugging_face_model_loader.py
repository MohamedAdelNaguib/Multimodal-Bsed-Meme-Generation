import os
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline


class HuggingFaceModelLoader:

    def __init__(self, task: str, model_name: str, model_quantization: bool = True):
        """
        Initialize the HuggingFaceModelLoader with optional model quantization.
        
        Args:
            task (str): The task for which the model is to be used.
            model_name (str): Identifier for the model to be loaded from Hugging Face.
            model_quantization (bool): If True, apply quantization to the model to reduce memory usage.
        
        Raises:
            ValueError: If the Hugging Face API token is not found in the environment variables.
        """
        hf_token = self._get_hf_token()
        
        if model_quantization:
            self.model = self._load_quantized_model(task, model_name, hf_token)
        else:
            self.model = self._load_model(task, model_name, hf_token)

    def _get_hf_token(self) -> str:
        """Retrieve the Hugging Face API token from environment variables."""
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        return hf_token

    def _load_quantized_model(self, task: str, model_name: str, hf_token: str):
        """
        Load a quantized model from Hugging Face.
        
        Args:
            task (str): The task for which the model is to be used.
            model_name (str): The model identifier.
            hf_token (str): The API token for Hugging Face.
        
        Returns:
            A Hugging Face pipeline object with quantization applied.
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        return pipeline(task=task, model=model_name, token=hf_token, model_kwargs={"quantization_config": quantization_config}, low_cpu_mem_usage=True)

    def _load_model(self, task: str, model_name: str, hf_token: str):
        """
        Load a model from Hugging Face without quantization.
        
        Args:
            task (str): The task for which the model is to be used.
            model_name (str): The model identifier.
            hf_token (str): The API token for Hugging Face.
        
        Returns:
            A Hugging Face pipeline object.
        """
        return pipeline(task=task, model=model_name, token=hf_token)
            